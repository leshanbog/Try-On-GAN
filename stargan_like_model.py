import torch
from torch import nn
import wandb
from torch.nn import functional as F

from utils import compute_gradient_penalty
    
    
class ResidualHiddenLayer(nn.Module):
    def __init__(self, n_inp, n_otpt, k, s, p):
        super().__init__()

        self.conv1 = nn.Conv2d(n_inp, n_inp, kernel_size=k, stride=s, padding=p, bias=False)
        self.conv2 = nn.Conv2d(n_inp, n_otpt, kernel_size=k, stride=s, padding=p, bias=False)
        self.expand = nn.Conv2d(n_inp, n_otpt, kernel_size=1, stride=s, padding=0, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out = self.act(self.conv1(x))
        out = self.conv2(x)

        out += self.expand(residual)

        return self.act(out)
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            ResidualHiddenLayer(3, 64, 4, 2, 1),
            ResidualHiddenLayer(64, 128, 4, 2, 1),
            ResidualHiddenLayer(128, 256, 4, 2, 1),
            ResidualHiddenLayer(256, 512, 4, 2, 1),
            ResidualHiddenLayer(512, 1024, 4, 2, 1),
            ResidualHiddenLayer(1024, 2048, 1, 1, 0),
        )

        self.predict_src = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.predict_match  = nn.Sequential(
            ResidualHiddenLayer(6, 16, 4, 2, 1),
            ResidualHiddenLayer(16, 32, 4, 2, 1),
            ResidualHiddenLayer(32, 64, 4, 2, 1),
            ResidualHiddenLayer(64, 128, 4, 2, 1),
            ResidualHiddenLayer(128, 256, 4, 2, 1),
            ResidualHiddenLayer(256, 512, 4, 2, 1),
            nn.Conv2d(512, 1, 1, 1, 0)
        )

    def forward(self, img, cc):
        assert img.shape == cc.shape
        p = self.predict_match(torch.cat((img, cc), dim=1))
        
        img = self.encode(img)
        f = torch.mean(self.predict_src(img), dim=(1,2,3))       

        return {
            'f': f,
            'p': p.squeeze()
        }

    
class TrueResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(256, affine=True)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(256, affine=True)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out = self.act(self.norm1(self.conv1(x)))

        out = self.norm2(self.conv2(out))

        out += residual

        return self.act(out)


class GeneratorUpscaleBlock(nn.Module):
    def __init__(self, n_inp, n_otpt, k, pd):
        super().__init__()
        self.conv = nn.Conv2d(n_inp, n_otpt, kernel_size=k, stride=1, padding=pd, bias=False)
        self.norm = nn.InstanceNorm2d(n_otpt, affine=True)
        self.act = nn.LeakyReLU(0.2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        return self.act(self.norm(self.conv(self.up(x))))


class GeneratorCore(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
            nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2),
            
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.bottleneck = nn.Sequential(
            TrueResidualBlock(),
            TrueResidualBlock(),
            TrueResidualBlock(),
            TrueResidualBlock(),
            TrueResidualBlock(),
            TrueResidualBlock(),
        )

        self.up = nn.Sequential(
            GeneratorUpscaleBlock(256, 128, 4, 2),
            GeneratorUpscaleBlock(128, 64, 4, 1),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.bottleneck(x)
        x = self.up(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = GeneratorCore()
        self.last = nn.Conv2d(64, 4, kernel_size=(8, 8), stride=(1, 1), padding=(3, 3), bias=False)

    def forward(self, img1, cc2, mask1=None):
        assert img1.shape == cc2.shape

        x = torch.cat((img1, cc2), dim=1)
        x = self.core(x)
        x = self.last(x)

        change = x[:, :3, :, :]
        mask = x[:, 3, :, :].unsqueeze(1)

        change = torch.tanh(change)
        mask = torch.sigmoid(mask)

        return {
            'out': change * mask + change * (1. - mask),
            'mask': mask
        }

class GeneratorGivenMask(nn.Module):
    def __init__(self):
        super().__init__()

        self.core = GeneratorCore()
        self.last = nn.Conv2d(64, 3, kernel_size=(8, 8), stride=(1, 1), padding=(3, 3), bias=False)

    def forward(self, img1, cc2, mask1):
        assert img1.shape == cc2.shape
        assert len(mask1.shape) == 4

        x = torch.cat((img1, cc2), dim=1)
        change = self.last(self.core(x))
        mask = mask1.to(dtype=torch.float)

        return {
            'out': change * mask + img1 * (1. - mask),
            'mask': mask
        }


class StarGAN:
    def __init__(self):
        if wandb.config.gt_given_mask:
            self.G = GeneratorGivenMask()
        else:
            self.G = Generator()

        self.D = Critic()

        self.optimizers = {
            'G': torch.optim.Adam(self.G.parameters(), wandb.config.generator_lr, betas=(0.5, 0.999), weight_decay=1e-4),
            'D': torch.optim.Adam(self.D.parameters(), wandb.config.critic_lr, betas=(0.5, 0.999), weight_decay=1e-4),
        }

        self.reconstruction_loss = nn.L1Loss()
        self.critic_step = 0


    def trainG(self, img1, mask1, cc1, cc2):
        if self.critic_step % wandb.config.critic_steps != 1:
            return

        self.optimizers['G'].zero_grad()

        g_out = self.G(img1, cc2, mask1)

        fake_img2 = g_out['out']
        fake_mask1 = g_out['mask']

        out_fake = self.D(fake_img2, cc2)

        back_g_out = self.G(fake_img2, cc1, mask1)
        back_gen_img1 = back_g_out['out']
        back_mask1 = back_g_out['mask']

        reconstruction_loss = self.reconstruction_loss(back_gen_img1, img1)
        wasserstein_loss = -torch.mean(out_fake['f'])
        match_loss = F.binary_cross_entropy_with_logits(out_fake['p'], torch.ones_like(out_fake['p']))

        correct_mask_loss = F.binary_cross_entropy(fake_mask1, mask1.to(dtype=torch.float)) + \
            F.binary_cross_entropy(back_mask1, mask1.to(dtype=torch.float))

        loss = 6 * reconstruction_loss + wasserstein_loss + 6 * match_loss + \
            correct_mask_loss * int(not wandb.config.gt_given_mask)

        loss.backward()

        self.optimizers['G'].step()

        return {
            'Reconstruction loss': reconstruction_loss.item(),
            'Wasserstein loss': wasserstein_loss.item(),
            'Match loss': match_loss.item(),
            'Correct mask loss': correct_mask_loss.item(),
            'Total loss': loss.item(),
        }


    def trainD(self, img1, mask1, cc1, cc2):
        self.optimizers['D'].zero_grad()

        fake_image = self.G(img1, cc2, mask1)['out'].detach()

        out_real = self.D(img1, cc1)
        out_fake = self.D(fake_image, cc2)

        wasserstein_loss = -torch.mean(out_real['f']) + torch.mean(out_fake['f'])
        gradient_penalty = compute_gradient_penalty(self.D, img1.data, fake_image.data, wandb.config.device)

        loss = wasserstein_loss + 10 * gradient_penalty

        result = {
            'Wasserstein loss': wasserstein_loss.item(),
            'Gradient Penalty loss': gradient_penalty.item(),    
        }

        if self.critic_step % wandb.config.critic_steps == 0:
            out_negative_probs = self.D(img1, cc2)['p']
            neg_match_loss = F.binary_cross_entropy_with_logits(out_negative_probs, torch.zeros_like(out_negative_probs))

            real_match_loss = F.binary_cross_entropy_with_logits(out_real['p'], torch.ones_like(out_real['p']))
            fake_match_loss = F.binary_cross_entropy_with_logits(out_fake['p'], torch.zeros_like(out_fake['p']))

            loss = loss + (real_match_loss + fake_match_loss + neg_match_loss) / 3

            result['Fake Match loss']  = fake_match_loss.item()
            result['Real Match loss'] = real_match_loss.item()
            result['Negative Match loss'] = neg_match_loss.item()
            result['Total loss'] = loss.item()
    
        loss.backward()

        self.optimizers['D'].step()
        self.critic_step += 1

        return result

    def generate(self, image, label):
        return self.G(image, label)

    def train(self):
        self.G.train()
        self.D.train()

    def eval(self):
        self.G.eval()
        self.D.eval()
        return self

    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        return self

    def get_state(self):
        return {
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'G_opt': self.optimizers['G'].state_dict(),
            'D_opt': self.optimizers['D'].state_dict(),
            'critic_step': self.critic_step,
        }

    def load_state_dict(self, state):
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])

        self.optimizers['G'].load_state_dict(state['G_opt'])
        self.optimizers['D'].load_state_dict(state['D_opt'])

        self.critic_step = state['critic_step']
