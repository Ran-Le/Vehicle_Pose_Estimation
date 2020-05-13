import numpy as np
from math import sin, cos
PATH = 'Dataset/'


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

