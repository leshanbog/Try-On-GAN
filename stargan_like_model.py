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

    def forward(self, x, label):
        assert x.shape == label.shape and x.shape[1:] == (3, 64, 64)
        p = self.predict_match(torch.cat((x, label), dim=1))
        
        x = self.encode(x)
        f = torch.mean(self.predict_src(x), dim=(1,2,3))       
        
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
    
    
    
class Generator(nn.Module):
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
            nn.Conv2d(64, 3, kernel_size=(8, 8), stride=(1, 1), padding=(3, 3), bias=False),
        )

    def forward(self, x, labels):
        assert x.shape == labels.shape and x.shape[1:] == (3, 64, 64)

        x = torch.cat((x, labels), dim=1)
        x = self.down(x)
        x = self.bottleneck(x)
        x = self.up(x)
        x = torch.tanh(x)
        return x


class StarGAN:
    def __init__(self):
        self.G = Generator()
        self.D = Critic()

        self.optimizers = {
            'G': torch.optim.Adam(self.G.parameters(), wandb.config.generator_lr, betas=(0.5, 0.999), weight_decay=1e-4),
            'D': torch.optim.Adam(self.D.parameters(), wandb.config.critic_lr, betas=(0.5, 0.999), weight_decay=1e-4),
        }

        self.reconstruction_loss = nn.L1Loss()
        self.critic_step = 0


    def trainG(self, img1, cc1, cc2):
        if self.critic_step % wandb.config.critic_steps != 1:
            return

        self.optimizers['G'].zero_grad()

        fake_img2 = self.G(img1, cc2)
        out_fake = self.D(fake_img2, cc2)

        back_gen_img1 = self.G(fake_img2, cc1)

        reconstruction_loss = self.reconstruction_loss(back_gen_img1, img1)
        wasserstein_loss = -torch.mean(out_fake['f'])
        match_loss = F.binary_cross_entropy_with_logits(out_fake['p'], torch.ones_like(out_fake['p']))

        loss = 10 * reconstruction_loss + wasserstein_loss + 3 * match_loss
        loss.backward()

        self.optimizers['G'].step()

        return {
            'Reconstruction loss': reconstruction_loss.item(),
            'Wasserstein loss': wasserstein_loss.item(),
            'Match loss': match_loss.item(),
            'Total loss': loss.item(),
        }


    def trainD(self, img1, cc1, cc2):
        self.optimizers['D'].zero_grad()

        fake_image = self.G(img1, cc2).detach()

        out_real = self.D(img1, cc1)
        out_fake = self.D(fake_image, cc2)

        wasserstein_loss = -torch.mean(out_real['f']) + torch.mean(out_fake['f'])
        gradient_penalty = compute_gradient_penalty(self.D, img1.data, fake_image.data, wandb.config.device)
        match_loss = F.binary_cross_entropy_with_logits(out_real['p'], torch.ones_like(out_real['p'])) + \
            F.binary_cross_entropy_with_logits(out_fake['p'], torch.zeros_like(out_fake['p']))

        loss = wasserstein_loss + 10 * gradient_penalty + match_loss

        loss.backward()

        self.optimizers['D'].step()
        self.critic_step += 1

        return {
            'Wasserstein loss': wasserstein_loss.item(),
            'Gradient Penalty loss': gradient_penalty.item(),
            'Match loss': match_loss.item(),
            'Total loss': loss.item(),
        }


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
