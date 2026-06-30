import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

# Dataset class for Coco style segmentation
class CocoSegDataset(Dataset):
    # Initialize the dataset
    def __init__(self, image_dir, json_path):
        self.image_dir = image_dir

        # Load annotations
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.images = data['images']
            self.annotations = data['annotations']

            # build index: image_id to annotations
            self.ann_map = {}
            for ann in self.annotations:
                img_id = ann['image_id']
                self.ann_map.setdefault(img_id, []).append(ann)

            self.id_to_image = {img['id']: img for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']

        file_name = img_info['file_name']
        # print("dataset.py - filename: ", file_name)

        img_path = os.path.join(self.image_dir, file_name)

        # Load image
        image = cv2.imread(img_path)
        # print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Get image size
        H, W = img_info['height'], img_info['width']

        # create empty mask. We will use this later for image segmentation
        mask = np.zeros((H,W), dtype=np.uint8)

        # draw polygons
        anns = self.ann_map.get(img_id, [])

        for ann in anns:
            for seg in ann["segmentation"]:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

        # turn them into tensors using PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0).float()

        # For debugging:
        # print("mask min:", mask.min().item())
        # print("mask max:", mask.max().item())
        # print("unique mask values:", torch.unique(mask))

        return image, mask
