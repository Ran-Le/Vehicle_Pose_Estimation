import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from efficientnet_pytorch import EfficientNet

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8


def mesh(batch, shape1, shape2):
    m1, m2 = np.meshgrid(np.linspace(0,1,shape2), np.linspace(0,1,shape1))
    m1 = np.tile(m1[None, None, :, :], [batch, 1, 1, 1]).astype('float32')
    m2 = np.tile(m2[None, None, :, :], [batch, 1, 1, 1]).astype('float32')
    return torch.cat(tuple([torch.tensor(m1), torch.tensor(m2)]), 1)


class cnn1(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )






