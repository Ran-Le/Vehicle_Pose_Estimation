import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualize import plt_cars
PATH = 'Dataset/'


def camera():
    # read the camera information and return its camera matrix
    f = open(PATH + 'camera/camera_intrinsic.txt', 'r')
    fx = float(f.readline().split()[2][:-1])
    fy = float(f.readline().split()[2][:-1])
    cx = float(f.readline().split()[2][:-1])
    cy = float(f.readline().split()[2][:-1])
    camera_mat = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
    return camera_mat


def train_data(filename):
    train = pd.read_csv(PATH + filename)
    return train



# test
idx = 2
data = train_data('train_small.csv')
cameraMat = camera()
# plt_car(cameraMat, data['PredictionString'][idx], data['ImageId'][idx])
image = plt.imread(PATH + 'train_images/' + data['ImageId'][idx] + '.jpg')
image = plt_cars(image, cameraMat, data['PredictionString'][idx])
plt.imshow(image)
plt.show()








