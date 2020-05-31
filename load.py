import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from visualize import plt_cars
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset
import time
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


def load_data(input, batch=4):
    camera_mat = camera()
    train_dir = PATH + 'train_images/'
    train, validate = train_test_split(input, test_size=0.3, random_state=13)
    train_data = ImageDataset(train, train_dir, camera_mat)
    validate_data = ImageDataset(validate, train_dir, camera_mat)
    train_loader = DataLoader(dataset=train_data, batch_size=batch, shuffle=True, num_workers=2)
    validate_loader = DataLoader(dataset=validate_data, batch_size=batch, shuffle=False, num_workers=0)
    return train_loader, validate_loader, validate_data, validate


def train_data_test(filename):
    train = pd.read_csv(PATH + filename)
    return train

if __name__ == "__main__":
    idx = 2
    t1 = time.time()
    data = train_data_test('train_small.csv')
    t2 = time.time()
    cameraMat = camera()
    # plt_car(cameraMat, data['PredictionString'][idx], data['ImageId'][idx])
    image = plt.imread(PATH + 'train_images/' + data['ImageId'][idx] + '.jpg')
    # image = plt_cars(image, cameraMat, data['PredictionString'][idx])
    plt.imshow(image)
    plt.show()

    # train, validate = load_data(data)
    # t3 = time.time()
    # print(train)
    # print(validate)
    # print(t2-t1)
    # print(t3-t2)







