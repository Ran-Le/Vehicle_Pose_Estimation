import numpy as np
import matplotlib.pyplot as plt
import cv2
from util import str2coords, coords2img, euler2mat

PATH = 'Dataset/'


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
    # color = (255, 0, 0)
    # pt1, pt2, pt3, pt4 = tuple(pts[0][:2]), tuple(pts[1][:2]), tuple(pts[2][:2]), tuple(pts[3][:2])
    # cv2.line(img, pt1, pt2, color, 10)
    # cv2.line(img, pt2, pt3, color, 10)
    # cv2.line(img, pt3, pt4, color, 10)
    # cv2.line(img, pt4, pt1, color, 10)

    color = (255, 0, 0)
    h = 3000
    top = []
    for pt in pts:
        top.append([pt[0], int(pt[1] - h / pt[2])])
    cv2.line(img, tuple(pts[0][:2]), tuple(pts[3][:2]), color, 6)
    cv2.line(img, tuple(pts[0][:2]), tuple(pts[1][:2]), color, 6)
    cv2.line(img, tuple(pts[1][:2]), tuple(pts[2][:2]), color, 6)
    cv2.line(img, tuple(pts[2][:2]), tuple(pts[3][:2]), color, 6)
    cv2.line(img, tuple(top[0][:]), tuple(top[3][:]), color, 6)
    cv2.line(img, tuple(top[0][:]), tuple(top[1][:]), color, 6)
    cv2.line(img, tuple(top[1][:]), tuple(top[2][:]), color, 6)
    cv2.line(img, tuple(top[2][:]), tuple(top[3][:]), color, 6)
    cv2.line(img, tuple(top[0][:]), tuple(pts[0][:2]), color, 6)
    cv2.line(img, tuple(pts[1][:2]), tuple(top[1][:]), color, 6)
    cv2.line(img, tuple(pts[2][:2]), tuple(top[2][:]), color, 6)
    cv2.line(img, tuple(pts[3][:2]), tuple(top[3][:]), color, 6)
    # draw the center
    color = (255, 255, 0)
    x, y, z = pts[-1, :]
    cv2.circle(img, (x, y), int(800 / z), color, -1)
    # for (p_x, p_y, p_z) in pts:
    #     cv2.circle(img, (p_x, p_y), int(800 / p_z), (255, 255, 0), -1)


def plt_car(camera_mat, coord_str, img_id):
    # plot the car center with red dot
    plt.figure()
    plt.imshow(cv2.imread(PATH + 'train_images/' + img_id + '.jpg'))
    plt.scatter(*coords2img(coord_str, camera_mat), color='red', s=50)
    plt.show()


def plt_cars(img, camera_mat, coord_str):
    coords = str2coords(coord_str)
    img = plt_cars_coords(img, camera_mat, coords)
    return img


def plt_cars_coords(img, camera_mat, coords):
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    p = np.array([[x_l, -y_l, -z_l, 1],
                  [x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, z_l, 1],
                  [-x_l, -y_l, -z_l, 1],
                  [0, 0, 0, 1]]).T
    img = img.copy()
    for coord in coords:
        mark_car(img, coord, p, camera_mat)
    return img