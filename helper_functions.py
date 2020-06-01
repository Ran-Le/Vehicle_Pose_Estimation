##########################################################################
# Load data
##########################################################################
import numpy as np
import cv2
from math import sin, cos

def label_to_list(s):
    '''
    Input:
        s: Label strings
    Output:
        list of dicts with keys from labels
    '''
    labels=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']
    res = []
    for i in np.array(s.split()).reshape([-1, 7]):
        res.append(dict(zip(labels, i.astype('float'))))
        if 'id' in res[-1]:
            res[-1]['id'] = int(res[-1]['id'])
    return res

def rotate(x, y):
    '''
    Input:
        angles to add
    Output:
        angles from -pi to pi
    '''
    x = x + y
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def get_img_coords(s):
    '''
    Input:
        s: Label strings
    Output:
        xs: img row
        ys: img col
    '''
    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

    coords = label_to_list(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_mat = np.dot(camera_matrix, P).T
    img_mat[:, 0] /= img_mat[:, 2]
    img_mat[:, 1] /= img_mat[:, 2]
    row = img_mat[:, 0]
    col = img_mat[:, 1]
    return row, col


def euler_to_rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_line(image, points):
    color = (255, 0, 0)
    h = 3000
    carN = len(points)
    top = []
    for point in points:
        top.append([point[0], int(point[1]-h/point[2])])
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 6)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 6)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 6)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(top[3][:]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(top[1][:]), color, 6)
    cv2.line(image, tuple(top[1][:]), tuple(top[2][:]), color, 6)
    cv2.line(image, tuple(top[2][:]), tuple(top[3][:]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(points[0][:2]), color, 6)
    cv2.line(image, tuple(points[1][:2]), tuple(top[1][:]), color, 6)
    cv2.line(image, tuple(points[2][:2]), tuple(top[2][:]), color, 6)
    cv2.line(image, tuple(points[3][:2]), tuple(top[3][:]), color, 6)
    return image


def draw_points(image, points):
    for (x, y, z) in points:
        cv2.circle(image, (x, y), int(800 / z), (255, 255, 0), -1)

    return image


def visualize(img, coords):

    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])

    return img

##########################################################################
# Image processing
##########################################################################
import numpy as np
import cv2
from scipy.optimize import minimize
from math import sin, cos


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

##########################################################################
# Generate dataset
##########################################################################
import torch
import numpy as np

from torch.utils.data import Dataset

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        img0 = cv2.imread(img_name)
        img = img_preprocess(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = get_mask_and_pose(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]
