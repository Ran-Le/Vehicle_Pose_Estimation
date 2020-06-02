##########################################################################
# Load data
##########################################################################
import numpy as np
import cv2
from math import sin, cos


def label_to_list(s):
    # s: label string
    # return list of dicts for each car
    labels = ['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']
    res = []
    for i in np.array(s.split()).reshape([-1, 7]):
        res.append(dict(zip(labels, i.astype('float'))))
        if 'id' in res[-1]:
            res[-1]['id'] = int(res[-1]['id'])
    return res


def rotate(x, y):
    # return an angle between -pi to pi
    x = x + y
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def get_img_coords(s):
    # from label string to img coordinate
    # read from txt file
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
    # from real world coordinate angle to image coordinate
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


def draw_line(image, pts):
    # plot the 3D box for a car in red
    color = (255, 0, 0)
    # tuning this number for car height
    h = 3000
    top = []
    for pt in pts:
        top.append([pt[0], int(pt[1]-h/pt[2])])
    cv2.line(image, tuple(pts[0][:2]), tuple(pts[3][:2]), color, 6)
    cv2.line(image, tuple(pts[0][:2]), tuple(pts[1][:2]), color, 6)
    cv2.line(image, tuple(pts[1][:2]), tuple(pts[2][:2]), color, 6)
    cv2.line(image, tuple(pts[2][:2]), tuple(pts[3][:2]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(top[3][:]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(top[1][:]), color, 6)
    cv2.line(image, tuple(top[1][:]), tuple(top[2][:]), color, 6)
    cv2.line(image, tuple(top[2][:]), tuple(top[3][:]), color, 6)
    cv2.line(image, tuple(top[0][:]), tuple(pts[0][:2]), color, 6)
    cv2.line(image, tuple(pts[1][:2]), tuple(top[1][:]), color, 6)
    cv2.line(image, tuple(pts[2][:2]), tuple(top[2][:]), color, 6)
    cv2.line(image, tuple(pts[3][:2]), tuple(top[3][:]), color, 6)
    return image


def draw_points(image, pts):
    # plot the center point of car in yellow
    for (x, y, z) in pts:
        cv2.circle(image, (x, y), int(800 / z), (255, 255, 0), -1)
    return image


def visualize(img, coords):
    # plot the car in the image with a box and a center point
    # tuning these numbers for a ideal car size
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)

    img = img.copy()
    for pt in coords:
        # Get values
        x, y, z = pt['x'], pt['y'], pt['z']
        yaw, pitch, roll = -pt['pitch'], -pt['yaw'], -pt['roll']
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
        img_cor_pts = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_pts = img_cor_pts.T
        img_cor_pts[:, 0] /= img_cor_pts[:, 2]
        img_cor_pts[:, 1] /= img_cor_pts[:, 2]
        img_cor_pts = img_cor_pts.astype(int)
        # plot
        img = draw_line(img, img_cor_pts)
        img = draw_points(img, img_cor_pts[-1:])

    return img