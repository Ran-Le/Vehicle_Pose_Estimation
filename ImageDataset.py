from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from ImgProcess import car_center


class ImageDataset(Dataset):
    def __init__(self, data, root, camera):
        self.data = data
        self.root = root
        self.camera = camera

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            idx = item.tolist()
        else:
            idx = item
        img_id, labels = self.data.to_numpy()[idx]
        img_name = self.root + img_id + '.jpg'
        img = cv2.imread(img_name)
        img = np.rollaxis(img, 2, 0)
        center, center_far = car_center(img, labels, self.camera)
        center_far = np.rollaxis(center_far, 2, 0)
        return [img, center, center_far]
