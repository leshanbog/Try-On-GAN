from PIL import Image
import numpy as np
import cv2
import json
import sys
import os
import matplotlib.pyplot as plt
import tqdm
import shutil


TARGET_CATEGORIES = [1, 2, 3, 4, 5, 6]
MAX_OCCLUSION = 1
MAX_ZOOM_IN = 2
OK_VIEWPOINT = [2]

split = sys.argv[1]  # train / validation
path = '../datasets/DeepFashion/' + split


def get_mask_from_polygon(polygons, img_path):
    mask = np.zeros(np.array(Image.open(img_path)).shape[:-1], dtype=np.uint8)
    for poly in polygons:
        pts = np.array([[poly[2 * i], poly[2 * i + 1]] 
                        for i in range(len(poly) // 2)], dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def get_centered_cloth(img_path, mask, bbox):
    img = np.array(Image.open(img_path))

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    m = max(w, h)
    
    f1, t1 = max(0, cy - m // 2), min(cy + m // 2, mask.shape[0])
    f2, t2 = max(0, cx - m // 2), min(cx + m // 2, mask.shape[1])
        
    if t1 - f1 != t2 - f2:
        if t1 - f1 > t2 - f2:
            if f2 == 0:
                t2 = t1 - f1
            else:
                f2 = mask.shape[1] - t1 + f1
        else:
            if f1 == 0:
                t1 = t2 - f2
            else:
                f1 = mask.shape[0]- t2 + f2

    cloth = img * mask[:, :, np.newaxis]
    center_cloth = cloth[f1: t1, f2: t2]
    center_cloth = cv2.resize(center_cloth, (64, 64))

    return center_cloth


shutil.rmtree(f'{path}/segmentation_masks', ignore_errors=True)
os.mkdir(f'{path}/segmentation_masks')

shutil.rmtree(f'{path}/centred_clothes', ignore_errors=True)
os.mkdir(f'{path}/centred_clothes')

total = len(os.listdir(f'{path}/annos'))


for annot_name in tqdm.tqdm(os.listdir(f'{path}/annos'), total=total):
    annot = json.load(open(f'{path}/annos/{annot_name}'))
    
    for k, v in annot.items():
        if k.startswith('item') and \
                v['occlusion'] <= MAX_OCCLUSION and \
                v['zoom_in'] <= MAX_ZOOM_IN and \
                v['viewpoint'] in OK_VIEWPOINT and \
                v['category_id'] in TARGET_CATEGORIES:
            obj_id = annot_name.split('.')[0]
            img_path = f'{path}/image/{obj_id}.jpg'

            try:
                mask = get_mask_from_polygon(v['segmentation'], img_path)
                
                if mask.sum() / np.prod(mask.shape) < 0.02:
                    continue
                
                cc = get_centered_cloth(img_path, mask, v['bounding_box'])

                np.save(f'{path}/segmentation_masks/{obj_id}_mask.npy', mask)
                np.save(f'{path}/centred_clothes/{obj_id}_cc.npy', cc)
            except KeyboardInterrupt:
                print('Stopping...')
                exit(0)
            except Exception:
                pass
