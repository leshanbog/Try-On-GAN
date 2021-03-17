import os

print(f'\n\ntaskset -p --cpu-list 40-50 {os.getpid()}\n\n')
_ = input('Done?')

import torch
import torchvision
import random
import numpy as np

import matplotlib.pyplot as plt
import tqdm

from calculate_fid import calculate_fid
from stargan_like_model import StarGAN
from custom_datasets import DeepFashionDataset


import wandb

@torch.no_grad()
def log_plots_and_fid():
    model.eval()
    fid = calculate_fid(val_dataloader, model.G)

    np.random.seed(34)
    img1, cc1, cc2 = next(iter(val_dataloader))

    perm = np.random.permutation(val_dataloader.batch_size)

    img1 = img1[perm].to(device)
    cc1 = cc1[perm].to(device)
    cc2 = cc2[perm].to(device)

    rows = 7
    fig, axs = plt.subplots(rows, 3, figsize=(15, 20))

    for i in range(rows):
        img = img1[i].unsqueeze(0)
        label = cc2[i].unsqueeze(0)

        fake_img = model.generate(img, label).detach()

        axs[i, 0].imshow((img.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 0].set_title('This person')
        axs[i, 0].axis('off')

        axs[i, 1].imshow((label.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 1].set_title('in that clothing')
        axs[i, 1].axis('off')

        axs[i, 2].imshow((fake_img.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 2].set_title('will look like this')
        axs[i, 2].axis('off')

    wandb.log({
        'FID': fid,
        'Epoch': epoch + 1,
        'Image translation examples': wandb.Image(fig),
    }, step=step)


wandb.login()
wandb.init(project='try-on-gan', id='x9wdsygj', resume=True);

wandb.config.critic_lr = 0.0001
wandb.config.generator_lr = 0.00015
wandb.config.critic_steps = 5
wandb.config.batch_size = 32
wandb.config.device = 'cuda:0'
wandb.config.max_epochs = 40
wandb.config.log_freq = 100


model = StarGAN().to(wandb.config.device)

state = torch.load('stargan_state.pt')
model.load_state_dict(state)

model.train()
wandb.watch(model.G)
wandb.watch(model.D)


transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

crop = torchvision.transforms.RandomResizedCrop((64, 64), scale=[0.8, 1.0], ratio=[0.9, 1.1])

augmented_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: crop(x) if random.random() < 0.75 else x),
    torchvision.transforms.Resize((64, 64)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = DeepFashionDataset('train', 
                                   photo_transform=augmented_transforms, 
                                   cloth_transform=augmented_transforms)

val_dataset = DeepFashionDataset('validation', 
                                 photo_transform=transforms, 
                                 cloth_transform=transforms)


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=wandb.config.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    num_workers=8)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=128,
    shuffle=False,
    drop_last=True,
    pin_memory=True,
    num_workers=8)


step = state['step']
device = wandb.config.device
start_epoch = state['epoch']
epoch = start_epoch

log_plots_and_fid()

for epoch in range(start_epoch, wandb.config.max_epochs):
    model.train()

    for img1, cc1, cc2 in tqdm.tqdm(train_dataloader, leave=False, total=len(train_dataloader)):
        img1 = img1.to(device)
        cc1 = cc1.to(device)
        cc2 = cc2.to(device)

        d_loss = model.trainD(img1, cc1, cc2)
        g_loss = model.trainG(img1, cc1, cc2)

        if step % wandb.config.log_freq == 0:
            wandb.log({
                'loss': {'Critic': d_loss, 'Generator': g_loss}
            }, step=step)

        step += 1

    state = model.get_state()
    state['epoch'] = epoch + 1
    state['step'] = step
    torch.save(state, './stargan_state.pt')

    log_plots_and_fid()
