##########################################################################
# Image processing
##########################################################################
import numpy as np
import cv2
from scipy.optimize import minimize
from math import sin, cos
from loading_functions import *


IMG_WIDTH = 1024
# IMG_WIDTH = (1024*2)
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8


def pose_preprocess(pose_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            pose_dict[k] = -pose_dict[k]
    for name in ['x', 'y', 'z']:
        pose_dict[name] = pose_dict[name] / 100
    pose_dict['roll'] = rotate(pose_dict['roll'], np.pi)
    pose_dict['pitch_sin'] = sin(pose_dict['pitch'])
    pose_dict['pitch_cos'] = cos(pose_dict['pitch'])
    pose_dict.pop('pitch')
    pose_dict.pop('id')
    return pose_dict


def pose_reverse(pose_dict):
    for name in ['x', 'y', 'z']:
        pose_dict[name] = pose_dict[name] * 100
    pose_dict['roll'] = rotate(pose_dict['roll'], -np.pi)

    pitch_sin = pose_dict[
        'pitch_sin'] / np.sqrt(pose_dict['pitch_sin']**2 + pose_dict['pitch_cos']**2)
    pitch_cos = pose_dict[
        'pitch_cos'] / np.sqrt(pose_dict['pitch_sin']**2 + pose_dict['pitch_cos']**2)
    pose_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return pose_dict


def img_preprocess(img, flip=False):
    img = img[img.shape[0] // 2:]
    side_padding = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    side_padding = side_padding[:, :img.shape[1] // 6]
    img = np.concatenate([side_padding, img, side_padding], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')


def get_mask_and_pose(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH //
                     MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    pose = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH //
                     MODEL_SCALE, 7], dtype='float32')
    coords = label_to_list(labels)
    xs, ys = get_img_coords(labels)
    for x, y, pose_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / \
            (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / \
            (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            pose_dict = pose_preprocess(pose_dict, flip)
            pose[x, y] = [pose_dict[n] for n in sorted(pose_dict)]
    if flip:
        mask = np.array(mask[:, ::-1])
        pose = np.array(pose[:, ::-1])
#     print(xs)
    return mask, pose

def convert_3d_to_2d(x, y, z, fx=2304.5479, fy=2305.8757, cx=1686.2379, cy=1354.9849):
    return x * fx / z + cx, y * fy / z + cy


def optimize_xy(xzy_slope,r, c, x0, y0, z0, flipped=False):
    IMG_SHAPE=(2710,3384,3)
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx, z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / \
            (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / \
            (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new


def remove_neighbors(coords,dist_thresh_clear=2):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < dist_thresh_clear:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def get_coord_from_pred(xzy_slope,prediction, flipped=False, threshold=0):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > threshold)
    col_names = sorted(
        ['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        pose_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(pose_reverse(pose_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = optimize_xy(xzy_slope,r, c,
                                                                        coords[-1]['x'],
                                                                        coords[-1]['y'],
                                                                        coords[-1]['z'], flipped)
    coords = remove_neighbors(coords)
    return coords


def coords_to_label(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

