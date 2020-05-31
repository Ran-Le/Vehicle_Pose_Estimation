import numpy as np
from math import sin, cos
from scipy.optimize import minimize
PATH = 'Dataset/'

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
IMG_SHAPE = (3384, 2710)
MODEL_SCALE = 8


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

DISTANCE_THRESH_CLEAR = 2

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    # stolen from https://www.kaggle.com/theshockwaverider/eda-visualization-baseline
    return x * fx / z + cx, y * fy / z + cy

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]


def optimize_xy(r, c, x0, y0, z0, slope):
    def distance_fn(xyz):
        x, y, z = xyz
        slope_err = (slope.predict([[x, z]])[0] - y) ** 2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x - r) ** 2 + (y - c) ** 2) + max(0.4, slope_err)

    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

def get_coords(pred, slope, threshold=0):
    logits = pred[0]
    regr_output = pred[1:]
    points = np.argwhere(logits > -0.5)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = \
            optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'], coords[-1]['z'], slope)
    coords = clear_duplicates(coords)
    return coords

def regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

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
        x = np.round(x).astype('int')
        # x = int(round(x))
        y = (ys[i] - img.shape[0] // 2) * IMG_HEIGHT / \
            (img.shape[0] // 2) / MODEL_SCALE
        y = np.round(y).astype('int')
        # y = int(round(y))
        if x >= 0 and x < modelWidth and y >= 0 and y < modelHeight:
            mask[y, x] = 1
            regr_dict = carinfo_cleanup(car_pose[i])
            #[pitch_cos,pitch_sin,roll,x,y,yaw,z]
            info[y, x] = [regr_dict[n] for n in sorted(regr_dict)]
    return mask, info
