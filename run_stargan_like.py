import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"

print(f'\n\ntaskset -p --cpu-list 10-30 {os.getpid()}\n\n')
_ = input('Done? ')


import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gt-given-mask', default=1, type=int)  # train with ground truth mask
    parser.add_argument('--max-epochs', default=70, type=int)
    parser.add_argument('--critic-lr', default=0.0001, type=float)
    parser.add_argument('--generator-lr', default=0.00015, type=float)
    parser.add_argument('--critic-steps', default=3, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--image-size', default=64, type=int)
    parser.add_argument('--dataset', default='DeepFashion', choices=('DeepFashion', 'VITON'))
    parser.add_argument('--log-freq', default=300, type=int)
    parser.add_argument('--resume', default='', type=str)  # path to checkpoint if we resume training
    parser.add_argument('--run-id', default='', type=str)  # wandb run ID if we resume training

    return parser.parse_args()


@torch.no_grad()
def log_plots_and_fid():
    model.eval()
    fid = calculate_fid(val_dataloader, model.G)

    np.random.seed(34)
    img1, mask1, cc1, cc2 = next(iter(val_dataloader))

    perm = np.random.permutation(val_dataloader.batch_size)

    img1 = img1[perm].to(device)
    mask1 = mask1[perm].to(device)
    cc2 = cc2[perm].to(device)

    rows = 7
    fig, axs = plt.subplots(rows, 3, figsize=(15, 20))

    for i in range(rows):
        im1 = img1[i].unsqueeze(0)
        c2 = cc2[i].unsqueeze(0)
        mas1 = mask1[i].unsqueeze(0)

        fake_img = model.G(im1, c2, mas1)['out'].detach()

        axs[i, 0].imshow((im1.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 0].set_title('This person')
        axs[i, 0].axis('off')

        axs[i, 1].imshow((c2.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 1].set_title('in that clothing')
        axs[i, 1].axis('off')

        axs[i, 2].imshow((fake_img.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
        axs[i, 2].set_title('will look like this')
        axs[i, 2].axis('off')

    result_log = {
        'FID': fid,
        'Epoch': epoch + 1,
        'Image translation examples': wandb.Image(fig),
    }

    if not wandb.config.gt_given_mask:
        fig, axs = plt.subplots(rows, 2, figsize=(15, 20))

        for i in range(rows):
            im1 = img1[i].unsqueeze(0)
            c2 = cc2[i].unsqueeze(0)
            mas1 = mask1[i].unsqueeze(0)

            fake_mask = model.G(im1, c2, mas1)['mask'].detach()

            axs[i, 0].imshow((im1.squeeze().permute(1, 2, 0).cpu() + 1) / 2)
            axs[i, 0].set_title('Person photo')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(fake_mask.squeeze().cpu())
            axs[i, 1].set_title('Predicted mask')
            axs[i, 1].axis('off')

        result_log['Predicted masks'] = wandb.Image(fig)

    wandb.log(result_log, step=step)

    
if __name__ == '__main__':
    args = parse_args()
    
    wandb.login()
    
    if args.run_id:
        wandb.init(project='try-on-gan', id=args.run_id, resume=True)
    else:
        wandb.init(project='try-on-gan')

    for p, v in vars(args).items():
        setattr(wandb.config, p, v)

    model = StarGAN().to(wandb.config.device)
    if wandb.config.gt_given_mask:
        print('Training with GT mask given')

    image_shape = (args.image_size, args.image_size)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    crop = torchvision.transforms.RandomResizedCrop(image_shape, scale=[0.8, 1.0], ratio=[0.9, 1.1])

    augmented_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda x: crop(x) if random.random() < 0.75 else x),
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print('Loading datasets...')

    if args.dataset == 'DeepFashion':
        train_dataset = DeepFashionDataset(
            'train', 
            image_shape,
            do_photo_augmentations=True, 
            cloth_transform=augmented_transforms
        )

        val_dataset = DeepFashionDataset(
            'validation',
            image_shape,
            do_photo_augmentations=False, 
            cloth_transform=transforms
        )
    elif args.dataset == 'VITON':
        raise NotImplementedError


    print('Making dataloaders...')

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

    device = wandb.config.device

    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state)
        step = state['step']
        epoch = state['epoch']

        log_plots_and_fid()
    else:
        step = 0
        epoch = 0

    wandb.watch(model.G, 'all')
    wandb.watch(model.D, 'all')

    print('Training')

    while epoch < wandb.config.max_epochs:
        model.train()

        for img1, mask1, cc1, cc2 in tqdm.tqdm(train_dataloader, leave=False, total=len(train_dataloader)):
            img1 = img1.to(device)
            mask1 = mask1.to(device)
            cc1 = cc1.to(device)
            cc2 = cc2.to(device)

            d_loss = model.trainD(img1, mask1, cc1, cc2)
            g_loss = model.trainG(img1, mask1, cc1, cc2)

            if step % wandb.config.log_freq == 0:
                wandb.log({
                    'loss': {'Critic': d_loss, 'Generator': g_loss}
                }, step=step)

            step += 1

        state = model.get_state()
        state['epoch'] = epoch + 1
        state['step'] = step
        
        suf = wandb.run.name
        torch.save(state, f'trained_models/stargan_state-{suf}.pt')

        log_plots_and_fid()
        epoch += 1
