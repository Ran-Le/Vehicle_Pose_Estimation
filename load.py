import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from math import sin, cos
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


def str2coords(s, names=('id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z')):
    # transfer string to 7 numbers for locations and orientations
    # return a list of dict
    coords = []
    for line in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, line.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


def coords2img(s, camera_mat):
    # use camera matrix to get the car location in image coordinates
    coords = str2coords(s)
    xs, ys, zs = [], [], []
    for coord in coords:
        xs.append(coord['x'])
        ys.append(coord['y'])
        zs.append(coord['z'])
    p = np.array(list(zip(xs, ys, zs))).T
    pos = camera_mat.dot(p).T
    img_x = pos[:, 0] / pos[:, 2]
    img_y = pos[:, 1] / pos[:, 2]
    return img_x, img_y


def euler2mat(yaw, pitch, roll):
    y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    p = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    r = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return y.dot(p.dot(r))


def mark_car(img, coord, p, camera_mat):
    # this function marks the car center with its angle info
    # reference math calculation
    x, y, z = coord['x'], coord['y'], coord['z']
    yaw, pitch, roll = -coord['pitch'], -coord['yaw'], -coord['roll']
    pos = np.array([x, y, z])
    Rt = np.eye(4)
    Rt[:3, 3] = pos
    Rt[:3, :3] = euler2mat(yaw, pitch, roll).T
    Rt = Rt[:3, :]
    pts = camera_mat.dot(Rt.dot(p))
    pts = pts.T
    pts[:, 0] /= pts[:, 2]
    pts[:, 1] /= pts[:, 2]
    pts = pts.astype(int)

    # draw the block
    color = (255, 0, 0)
    pt1, pt2, pt3, pt4 = tuple(pts[0][:2]), tuple(pts[1][:2]), tuple(pts[2][:2]), tuple(pts[3][:2])
    cv2.line(img, pt1, pt2, color, 10)
    cv2.line(img, pt2, pt3, color, 10)
    cv2.line(img, pt3, pt4, color, 10)
    cv2.line(img, pt4, pt1, color, 10)

    # draw the center
    color = (0, 255, 0)
    x, y, z = pts[-1, :]
    cv2.circle(img, (x, y), int(1000 / z), color, -1)


def plt_car(camera_mat, coord_str, img_id):
    # plot the car center with red dot
    plt.figure()
    plt.imshow(plt.imread(PATH + 'train_images/' + img_id + '.jpg'))
    plt.scatter(*coords2img(coord_str, camera_mat), color='red', s=50)
    plt.show()


def plt_cars(img, camera_mat, coord_str):
    # plt all cars in the image
    # how to find the following params?
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    p = np.array([[x_l, -y_l, -z_l, 1],
                  [x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, -z_l, 1],
                  [0, 0, 0, 1]]).T
    img = img.copy()
    coords = str2coords(coord_str)
    for coord in coords:
        mark_car(img, coord, p, camera_mat)
    return img


# test
idx = 0
data = train_data('train_small.csv')
cameraMat = camera()
# plt_car(cameraMat, data['PredictionString'][idx], data['ImageId'][idx])
image = plt.imread(PATH + 'train_images/' + data['ImageId'][idx] + '.jpg')
image = plt_cars(image, cameraMat, data['PredictionString'][idx])
plt.imshow(image)
plt.show()








