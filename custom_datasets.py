import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import json
import random
from collections import defaultdict
import torchvision.transforms.functional as TF



TARGET_CATEGORIES = [1, 2, 3, 4, 5, 6]  # mutually exclusive


class DeepFashionDataset(Dataset):
    def __init__(self, split, do_photo_augmentations=False, cloth_transform=None):
        assert isinstance(do_photo_augmentations, bool)

        self.path = f'../datasets/DeepFashion/{split}'
        
        self.do_photo_augmentations = do_photo_augmentations
        self.cloth_transform = cloth_transform
        
        self.objects = [x[:6] for x in os.listdir(f'{self.path}/centred_clothes')]
        
        self.idx_to_cat = np.zeros(len(self.objects))
        self.cat_to_idx = defaultdict(list)
        
        for i, oid in enumerate(self.objects):
            annot = json.load(open(f'{self.path}/annos/{oid}.json'))
            cat = self._get_cat_from_annot(annot)
            self.idx_to_cat[i] = cat
            self.cat_to_idx[cat].append(i)
            
    def _get_cat_from_annot(self, annot):
        for k, v in annot.items():
            if k.startswith('item') and v['category_id'] in TARGET_CATEGORIES:
                return v['category_id']
        
    def __len__(self):
        return len(self.objects)
    
    def _get_centred_cloth(self, idx):
        centred_cloth = Image.fromarray(np.load(f'{self.path}/centred_clothes/{self.objects[idx]}_cc.npy'))
        if self.cloth_transform:
            centred_cloth = self.cloth_transform(centred_cloth)
        return centred_cloth

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        annot = json.load(open(f'{self.path}/annos/{self.objects[idx]}.json'))
        cat = self._get_cat_from_annot(annot)

        person_image = Image.open(f'{self.path}/image/{self.objects[idx]}.jpg')
        mask = torch.BoolTensor(np.load(f'{self.path}/segmentation_masks/{self.objects[idx]}_mask.npy')).unsqueeze(0)
        
        assert mask.shape[1:] == (person_image.height, person_image.width)
        
        if self.do_photo_augmentations:
            if random.random() < 0.75:
                # Random crop
                i, j, h, w = transforms.RandomResizedCrop.get_params(person_image, scale=[0.8, 1.0], ratio=[0.9, 1.1])
                person_image = TF.resized_crop(person_image, i, j, h, w, size=(64, 64))
                mask = TF.resized_crop(mask, i, j, h, w, size=(64, 64))

            # Resize
            person_image = TF.resize(person_image, size=(64, 64))
            mask = TF.resize(mask, size=(64, 64))

            # Random horizontal flipping
            if random.random() > 0.5:
                person_image = TF.hflip(person_image)
                mask = TF.hflip(mask)

        person_image = TF.normalize(TF.to_tensor(person_image), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        rand_index = np.random.choice(len(self.cat_to_idx[cat]))
            
        return person_image, \
            mask, \
            self._get_centred_cloth(idx), \
            self._get_centred_cloth(self.cat_to_idx[cat][rand_index])
