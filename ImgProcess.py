import numpy as np
from util import str2coords, coords2img
from math import sin, cos

# Compressed image info
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


def carinfo_cleanup(pose):
    """
    scale down x, y, z to 1/100
    roll angle plus pi
    From:
    [id,x,y,z,yaw,pitch,roll]
    To:
    [x,y,z,yaw,pitch_sin,pitch_cos,roll]
    """
    for i in ['x', 'y', 'z']:
        pose[i] = pose[i] / 100
    pose['roll'] = rotate(pose['roll'], np.pi)
    pose['pitch_sin'] = sin(pose['pitch'])
    pose['pitch_cos'] = cos(pose['pitch'])
    pose.pop('pitch')
    pose.pop('id')
    return pose


def car_center(img, labels, camera):
    """
    Input:
    image, labels, camera info
    Output:
    mask matrix, pose info matrix (7 layers)
    """
    modelHeight = IMG_HEIGHT // MODEL_SCALE
    modelWidth = IMG_WIDTH // MODEL_SCALE
    mask = np.zeros([modelHeight, modelWidth], dtype='float32')
    # regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    info = np.zeros([modelHeight, modelWidth, 7], dtype='float32')
    car_pose = str2coords(labels)
    xs, ys = coords2img(labels, camera)
    for i in range(len(car_pose)):
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
            regr_dict = carinfo_cleanup(car_pose[i])
            #[pitch_cos,pitch_sin,roll,x,y,yaw,z]
            info[y, x] = [regr_dict[n] for n in sorted(regr_dict)]
    return mask, info
