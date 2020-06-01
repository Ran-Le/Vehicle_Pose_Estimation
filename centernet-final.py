import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

PATH = './Dataset/'
os.listdir(PATH)

train = pd.read_csv(PATH + 'train.csv',nrows=400)
# train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')

drop_images = ['ID_1a5a10365', 'ID_4d238ae90.jpg', 'ID_408f58e9f', 'ID_bb1d991f6', 'ID_c44983aeb'] 
train = train[~train['ImageId'].isin(drop_images)]

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

train.head()

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

img = imread(PATH + 'train_images/ID_8a6e65317' + '.jpg')
IMG_SHAPE = img.shape

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

inp = train['PredictionString'][0]
print('Example input:\n', inp)
print()
print('Output:\n', str2coords(inp))



def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x


def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys



from math import sin, cos

def euler_to_Rot(yaw, pitch, roll):
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
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image

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
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(800 / p_z), (255, 255, 0), -1)

    return image




def visualize(img, coords):

    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
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




IMG_WIDTH = 1024
# IMG_WIDTH = (1024*2)
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8

def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
#     print(xs)
    return mask, regr

DISTANCE_THRESH_CLEAR = 2

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0, flipped=False):
    def distance_fn(xyz):
        x, y, z = xyz
        xx = -x if flipped else x
        slope_err = (xzy_slope.predict([[xx,z]])[0] - y)**2
        x, y = convert_3d_to_2d(x, y, z)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        y = (y + IMG_SHAPE[1] // 6) * IMG_WIDTH / (IMG_SHAPE[1] * 4 / 3) / MODEL_SCALE
        return max(0.2, (x-r)**2 + (y-c)**2) + max(0.4, slope_err)
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z_new

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

def extract_coords(prediction, flipped=False,threshold=0):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > threshold)
    col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
    coords = []
    for r, c in points:
        regr_dict = dict(zip(col_names, regr_output[:, r, c]))
        coords.append(_regr_back(regr_dict))
        coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
        coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] =                 optimize_xy(r, c,
                            coords[-1]['x'],
                            coords[-1]['y'],
                            coords[-1]['z'], flipped)
    coords = clear_duplicates(coords)
    return coords

def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)



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
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)
        
        return [img, mask, regr]



train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

# df_train, df_test = train_test_split(train, test_size=0.02, random_state=231)
# df_train, df_dev = train_test_split(df_train, test_size=0.02, random_state=231)

df_train, df_dev = train_test_split(train, test_size=0.01, random_state=231)
df_test = test


train_dataset = CarDataset(df_train, train_images_dir, training=True)
dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
# test_dataset = CarDataset(df_test, train_images_dir, training=False)
test_dataset = CarDataset(df_test, test_images_dir, training=False)

idx,label=train_dataset.df.to_numpy()[0]


# BATCH_SIZE = 1
BATCH_SIZE = 4

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)




from efficientnet_pytorch import EfficientNet

effnet_ver = 'b0'
dropout_rate = 0.3
# dropout_rate = 0.0

def set_dropout(model, drop_rate):
    # source: https://discuss.pytorch.org/t/how-to-increase-dropout-rate-during-training/58107/4
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
            print("name:", name)
            print("children:\n", child)

def effnet_dropout(drop_rate):
    base_model0 = EfficientNet.from_pretrained(f"efficientnet-{effnet_ver}")
    set_dropout(base_model0, drop_rate)
    return base_model0

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    '''in_ch=>out_ch,dim_out==dim_in '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class res_block(nn.Module):
    '''(conv => ReLU)*3 + 1*1Conv+BN '''
    '''in_ch=>out_ch,dim_out==dim_in '''
    def __init__(self, in_ch,out_ch):
        super(res_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//8, 3, padding=1),
            nn.BatchNorm2d(out_ch//8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch//8, out_ch//8, 3, padding=1),
            nn.BatchNorm2d(out_ch//8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch//8, out_ch//4, 3, padding=1),
            nn.BatchNorm2d(out_ch//4),
            nn.ReLU(inplace=True)
        )
        self.bn=nn.BatchNorm2d(out_ch//2)
        self.sc=nn.Conv2d(in_ch,out_ch//2,1)
        self.addconv=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x_shortcut = self.sc(x)
        x_p1 = self.conv1(x)
        x_p2 = self.conv2(x_p1)
        x_p3 = self.conv3(x_p2)
        x_path=torch.cat([x_p1, x_p2, x_p3], dim=1)
        x_path=self.bn(x_path)
        x_out=torch.cat([x_shortcut, x_path], dim=1)
        x_out=self.addconv(x_out)
        return x_out

class respath_block(nn.Module):
    '''3*3Conv + 1*1Conv '''
    '''in_ch=>out_ch,dim_out==dim_in '''
    def __init__(self, in_ch,out_ch):
        super(respath_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 3, padding=1),
            nn.BatchNorm2d(out_ch//2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 1),
            nn.BatchNorm2d(out_ch//2)
        )
        self.addconv=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x_p1 = self.conv1(x)
        x_p2 = self.conv2(x)
        x_path=torch.cat([x_p1, x_p2], dim=1)
        x_out=self.addconv(x_path)
        return x_out

class res_path(nn.Module):
    '''(respath_block)*4 '''
    '''in_ch=>out_ch,dim_out==dim_in '''
    def __init__(self, in_ch,out_ch):
        super(res_path, self).__init__()
#         self.rp1=res_block(in_ch,out_ch)
#         self.rp2=res_block(out_ch,out_ch)
#         self.rp3=res_block(out_ch,out_ch)
#         self.rp4=res_block(out_ch,out_ch)
        self.rp1=respath_block(in_ch,out_ch)
        self.rp2=respath_block(out_ch,out_ch)
        self.rp3=res_block(out_ch,out_ch)
        self.rp4=res_block(out_ch,out_ch)


    def forward(self, x):
        x=self.rp1(x)
        x=self.rp2(x)
        x=self.rp3(x)
        x=self.rp4(x)
        return x
    
class up(nn.Module):
    '''(Respath+ConvT)=>ResBlock '''
    '''in_ch1(ConvT),in_ch2(Respath)=>out_ch,dim_out==2*dim_in '''
    def __init__(self, in_ch1,in_ch2, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch1, in_ch1, 2,stride=2,padding=1,output_padding=1)
        self.bn=nn.BatchNorm2d(in_ch1)
        self.relu=nn.ReLU()
        
        self.respath=res_path(in_ch2,2*in_ch2)
        
        self.conv = double_conv(in_ch1+2*in_ch2, out_ch)
#         self.conv = res_block(in_ch1+2*in_ch2, out_ch)
        
    def forward(self, x1, x2=None):
        x1=self.up(x1)
        x1=self.bn(x1)
        x1=self.relu(x1)
        
        x2=self.respath(x2)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
            
        x = self.conv(x)
        return x

class output_conv(nn.Module):
    '''(conv => BN => ReLU => 1*1conv) '''
    '''in_ch=>out_ch,dim_out==dim_in '''
    def __init__(self, in_ch,h_ch, out_ch):
        super(output_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, h_ch, 3, padding=1),
            nn.BatchNorm2d(h_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class MyUNet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.drop_rate = dropout_rate
        self.base_model = EfficientNet.from_pretrained(f"efficientnet-{effnet_ver}")
#         self.base_model = effnet_dropout(drop_rate = self.drop_rate)
        self.conv0 = double_conv(3, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
#         self.conv0 = res_block(3, 64)
#         self.conv1 = res_block(64, 128)
#         self.conv2 = res_block(128, 512)
#         self.conv3 = res_block(512, 1024)
        self.mp = nn.MaxPool2d(2)
        
        if effnet_ver == 'b0': self.up1 = up(1280, 1024, 512)
        elif effnet_ver == 'b1': self.up1 = up(1280, 1024, 512)
        elif effnet_ver == 'b2': self.up1 = up(1408, 1024, 512)
        elif effnet_ver == 'b3': self.up1 = up(1536, 1024, 512)
        elif effnet_ver == 'b4': self.up1 = up(1792, 1024, 512)
        elif effnet_ver == 'b5': self.up1 = up(2048, 1024, 512)
#         self.up1 = up(1536,1024, 512)
        self.up2 = up(512,512, 256)
#         self.outc = nn.Conv2d(256, n_classes, 1)
        self.poseconv=output_conv(256,1024,7)
        self.detectionconv = output_conv(256,256, 1)

    def forward(self, x):
        # torch.Size([1, 3, 320, 1024])
        batch_size = x.shape[0]

        x1 = self.mp(self.conv0(x))
        # torch.Size([1, 64, 160, 512])
        x2 = self.mp(self.conv1(x1))
        # torch.Size([1, 128, 80, 256])
        x3 = self.mp(self.conv2(x2))
        # torch.Size([1, 512, 40, 128])
        x4 = self.mp(self.conv3(x3))
        # torch.Size([1, 1024, 20, 64])
        
        feats = self.base_model.extract_features(x)
        
        x = self.up1(feats, x4)
# torch.Size([1, 512, 20, 64])
        x = self.up2(x, x3)
# torch.Size([1, 256, 40, 128])

        xout_1 = self.detectionconv(x)

        xout_2 = self.poseconv(x)

        xout = torch.cat([xout_1, xout_2], dim=1)
#         xout=self.outc(x)

# torch.Size([1, 8, 40, 128])
        return xout



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)

# n_epochs = 10
n_epochs = 2

model = MyUNet(8).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

# model = torch.load('./model_test.pth')



def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     plt.imshow(prediction[0, 0].data.cpu().numpy())
#     plt.show()
#     print(prediction.shape)
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    temp=torch.abs(pred_regr - regr)
#     temp=(pred_regr - regr)**2
#     temp=torch.sqrt(temp)
    regr_loss = (temp.sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
    
    gamma=5.0
    # Sum
    loss = mask_loss + gamma*regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return mask_loss,regr_loss,loss


def train_model(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        mask_loss,regr_loss,loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_mask_loss'] = mask_loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_regr_loss'] = regr_loss.data.cpu().numpy()
        
        loss.backward()
        
        optimizer.step()
        exp_lr_scheduler.step()
    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))
    print('Train mask loss: {:.4f}'.format(mask_loss))
    print('Train regr loss: {:.4f}'.format(regr_loss))

def evaluate_model(epoch, history=None):
    model.eval()
    loss = 0
    mask_loss=0
    regr_loss=0
    
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            mask_loss_t,regr_loss_t,loss_t = criterion(output, mask_batch, regr_batch, size_average=False)
            mask_loss+=mask_loss_t
            regr_loss+=regr_loss_t
            loss+=loss_t
    
    loss /= len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'dev_mask_loss'] = mask_loss.cpu().numpy()
        history.loc[epoch, 'dev_regr_loss'] = regr_loss.cpu().numpy()
    
    print('Dev loss: {:.4f}'.format(loss))
    print('Dev mask loss: {:.4f}'.format(mask_loss))
    print('Dev regr loss: {:.4f}'.format(regr_loss))
    
# def test_model(epoch, history=None):
#     model.eval()
#     loss = 0
    
#     with torch.no_grad():
#         for img_batch, mask_batch, regr_batch in test_loader:
#             img_batch = img_batch.to(device)
#             mask_batch = mask_batch.to(device)
#             regr_batch = regr_batch.to(device)

#             output = model(img_batch)

#             mask_loss,regr_loss,loss += criterion(output, mask_batch, regr_batch, size_average=False).data
    
#     loss /= len(test_loader.dataset)
    
#     if history is not None:
#         history.loc[epoch, 'test_loss'] = loss.cpu().numpy()
    
#     print('Test loss: {:.4f}'.format(loss))
#     print('Test mask loss: {:.4f}'.format(mask_loss))
#     print('Test regr loss: {:.4f}'.format(regr_loss))




import gc

history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history)
    evaluate_model(epoch, history)

# torch.save(model.state_dict(), './model_test.pth')
# torch.save(model, './model_test.pth')



series = history.dropna()['dev_loss']


points_df = pd.DataFrame()
for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
    arr = []
    for ps in train['PredictionString']:
        coords = str2coords(ps)
        arr += [c[col] for c in coords]
    points_df[col] = arr

zy_slope = LinearRegression()
X = points_df[['z']]
y = points_df['y']
zy_slope.fit(X, y)
print('MAE without x:', mean_absolute_error(y, zy_slope.predict(X)))



xzy_slope = LinearRegression()
X = points_df[['x', 'z']]
y = points_df['y']
xzy_slope.fit(X, y)
print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))

print('\ndy/dx = {:.3f}\ndy/dz = {:.3f}'.format(*xzy_slope.coef_))




import gc
torch.cuda.empty_cache()
gc.collect()




torch.cuda.empty_cache()
predictions = []

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model.eval()

for img, _, _ in tqdm(test_loader):
    with torch.no_grad():
        output = model(img.to(device))
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out,threshold=0)
        s = coords2str(coords)
        predictions.append(s)


# test = pd.read_csv(PATH + 'sample_submission.csv')
# test['PredictionString'] = predictions
# test.to_csv('predictions.csv', index=False)
# test.head()
