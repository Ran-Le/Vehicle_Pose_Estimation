import numpy as np
from util import str2coords, coords2img
from math import sin, cos

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8


def rotate(x, dx):
    """
    rotate angle and map to [-pi, pi]
    """
    x = x + dx
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def carinfo_cleanup(regr_dict):
    """
    scale down x, y, z to 1/100
    roll angle plus pi
    From:
    [x,y,z,yaw,pitch,roll,id]
    To:
    [x,y,z,yaw,pitch_sin,pitch_cos,roll]
    """
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def car_center(img, labels, camera):
    """
    Input:
    image, labels, camera info
    Output:
    mask matrix, corresponding info matrix (7 layers)
    """
    modelHeight = IMG_HEIGHT // MODEL_SCALE
    modelWidth = IMG_WIDTH // MODEL_SCALE
    mask = np.zeros([modelHeight, modelWidth], dtype='float32')
    # regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    info = np.zeros([modelHeight, modelWidth, 7], dtype='float32')
    car_pos = str2coords(labels)
    xs, ys = coords2img(labels, camera)
    for i in range(len(car_pos)):
        x = (xs[i] + img.shape[1] // 6) * IMG_WIDTH / \
            (img.shape[1] * 4/3) / MODEL_SCALE
        # x = np.round(x).astype('int')
        x = int(round(x))
        y = (ys[i] - img.shape[0] // 2) * IMG_HEIGHT / \
            (img.shape[0] // 2) / MODEL_SCALE
        # y = np.round(y).astype('int')
        y = int(round(y))
        if x >= 0 and x < modelWidth and y >= 0 and y < modelHeight:
            mask[y, x] = 1
            regr_dict = carinfo_cleanup(car_pos[i])
            info[y, x] = [regr_dict[n] for n in sorted(regr_dict)]
    return mask, info
