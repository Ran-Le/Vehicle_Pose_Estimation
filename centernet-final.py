from helper_functions import *



##########################################################################
# Setup model
##########################################################################
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F

EFFNET_VER = 'b0'
DROPOUT_RATE = 0.3
# DROPOUT_RATE = 0.0


def set_dropout(model, drop_rate):
    # source:
    # https://discuss.pytorch.org/t/how-to-increase-dropout-rate-during-training/58107/4
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
            print("name:", name)
            print("children:\n", child)


def effnet_dropout(drop_rate):
    base_model0 = EfficientNet.from_pretrained(f"efficientnet-{EFFNET_VER}")
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

    def __init__(self, in_ch, out_ch):
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
        self.bn = nn.BatchNorm2d(out_ch//2)
        self.sc = nn.Conv2d(in_ch, out_ch//2, 1)
        self.addconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x_shortcut = self.sc(x)
        x_p1 = self.conv1(x)
        x_p2 = self.conv2(x_p1)
        x_p3 = self.conv3(x_p2)
        x_path = torch.cat([x_p1, x_p2, x_p3], dim=1)
        x_path = self.bn(x_path)
        x_out = torch.cat([x_shortcut, x_path], dim=1)
        x_out = self.addconv(x_out)
        return x_out


class respath_block(nn.Module):
    '''3*3Conv + 1*1Conv '''
    '''in_ch=>out_ch,dim_out==dim_in '''

    def __init__(self, in_ch, out_ch):
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
        self.addconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x_p1 = self.conv1(x)
        x_p2 = self.conv2(x)
        x_path = torch.cat([x_p1, x_p2], dim=1)
        x_out = self.addconv(x_path)
        return x_out


class res_path(nn.Module):
    '''(respath_block)*4 '''
    '''in_ch=>out_ch,dim_out==dim_in '''

    def __init__(self, in_ch, out_ch):
        super(res_path, self).__init__()
#         self.rp1=res_block(in_ch,out_ch)
#         self.rp2=res_block(out_ch,out_ch)
#         self.rp3=res_block(out_ch,out_ch)
#         self.rp4=res_block(out_ch,out_ch)
        self.rp1 = respath_block(in_ch, out_ch)
        self.rp2 = respath_block(out_ch, out_ch)
        self.rp3 = res_block(out_ch, out_ch)
        self.rp4 = res_block(out_ch, out_ch)

    def forward(self, x):
        x = self.rp1(x)
        x = self.rp2(x)
        x = self.rp3(x)
        x = self.rp4(x)
        return x


class up_sampling(nn.Module):
    '''(Respath+ConvT)=>ResBlock '''
    '''in_ch1(ConvT),in_ch2(Respath)=>out_ch,dim_out==2*dim_in '''

    def __init__(self, in_ch1, in_ch2, out_ch):
        super(up_sampling, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_ch1, in_ch1, 2, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(in_ch1)
        self.relu = nn.ReLU()

        self.respath = res_path(in_ch2, 2*in_ch2)

        self.conv = double_conv(in_ch1+2*in_ch2, out_ch)
#         self.conv = res_block(in_ch1+2*in_ch2, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        x2 = self.respath(x2)

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

    def __init__(self, in_ch, h_ch, out_ch):
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


class ConvMultiRes(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(ConvMultiRes, self).__init__()
        self.drop_rate = DROPOUT_RATE
        self.base_model = EfficientNet.from_pretrained(f"efficientnet-{EFFNET_VER}")
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

        if EFFNET_VER == 'b0':
            self.up1 = up_sampling(1280, 1024, 512)
        elif EFFNET_VER == 'b1':
            self.up1 = up_sampling(1280, 1024, 512)
        elif EFFNET_VER == 'b2':
            self.up1 = up_sampling(1408, 1024, 512)
        elif EFFNET_VER == 'b3':
            self.up1 = up_sampling(1536, 1024, 512)
        elif EFFNET_VER == 'b4':
            self.up1 = up_sampling(1792, 1024, 512)
        elif EFFNET_VER == 'b5':
            self.up1 = up_sampling(2048, 1024, 512)
#         self.up1 = up_sampling(1536,1024, 512)
        self.up2 = up_sampling(512, 512, 256)
#         self.outc = nn.Conv2d(256, n_classes, 1)
        self.poseconv = output_conv(256, 1024, 7)
        self.detectionconv = output_conv(256, 256, 1)

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


##########################################################################
# Loss
##########################################################################
import torch
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def criterion(prediction, mask, regr, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
#     plt.imshow(prediction[0, 0].data.cpu().numpy())
#     plt.show()
#     print(prediction.shape)
#     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    mask_loss = mask * torch.log(pred_mask + 1e-12) + \
        (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()

    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    temp = torch.abs(pred_regr - regr)
#     temp=(pred_regr - regr)**2
#     temp=torch.sqrt(temp)
    regr_loss = (temp.sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    gamma = 5.0
    # Sum
    loss = mask_loss + gamma*regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return mask_loss, regr_loss, loss


def train_model(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        mask_loss, regr_loss, loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx /
                        len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx /
                        len(train_loader), 'train_mask_loss'] = mask_loss.data.cpu().numpy()
            history.loc[epoch + batch_idx /
                        len(train_loader), 'train_regr_loss'] = regr_loss.data.cpu().numpy()

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
    mask_loss = 0
    regr_loss = 0

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            mask_loss_t, regr_loss_t, loss_t = criterion(
                output, mask_batch, regr_batch, size_average=False)
            mask_loss += mask_loss_t
            regr_loss += regr_loss_t
            loss += loss_t

    loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'dev_mask_loss'] = mask_loss.cpu().numpy()
        history.loc[epoch, 'dev_regr_loss'] = regr_loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))
    print('Dev mask loss: {:.4f}'.format(mask_loss))
    print('Dev regr loss: {:.4f}'.format(regr_loss))






















##########################################################################
# Training
##########################################################################
import pandas as pd
import torch
import gc


import torch
import torch.optim as optim
from torch.optim import lr_scheduler


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

PATH = './Dataset/'
os.listdir(PATH)

debugging_mode=True
if debugging_mode:
    train = pd.read_csv(PATH + 'train.csv', nrows=20)
    test = pd.read_csv(PATH + 'sample_submission.csv',nrows=2)
else:
    train = pd.read_csv(PATH + 'train.csv')
    test = pd.read_csv(PATH + 'sample_submission.csv')

# Remove damaged images from the dataset
img_damaged = ['ID_1a5a10365', 'ID_4d238ae90.jpg',
               'ID_408f58e9f', 'ID_bb1d991f6', 'ID_c44983aeb']
train = train[~train['ImageId'].isin(img_damaged)]

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

idx, label = train_dataset.df.to_numpy()[0]


# BATCH_SIZE = 1
BATCH_SIZE = 4

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset,
                        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# print(device)

# n_epochs = 10
n_epochs = 2

model = ConvMultiRes(8).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(
    n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

# model = torch.load('./model_test.pth')


history = pd.DataFrame()

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train_model(epoch, history)
    evaluate_model(epoch, history)

##########################################################################
# Save model
##########################################################################
import torch
import gc
import pandas as pd
from sklearn.linear_model import LinearRegression

save_model = False
make_predictions = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if save_model:
    torch.save(model, './model_test_org.pth')

if make_predictions:

    points_df = pd.DataFrame()
    for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
        arr = []
        for ps in train['PredictionString']:
            coords = label_to_list(ps)
            arr += [c[col] for c in coords]
        points_df[col] = arr

    zy_slope = LinearRegression()
    X = points_df[['z']]
    y = points_df['y']
    zy_slope.fit(X, y)

    # Will use this model later
    xzy_slope = LinearRegression()
    X = points_df[['x', 'z']]
    y = points_df['y']
    xzy_slope.fit(X, y)

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.empty_cache()
    predictions = []

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model.eval()

    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = get_coord_from_pred(out, threshold=0)
            s = coords_to_label(coords)
            predictions.append(s)

    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv('predictions_org.csv', index=False)
    test.head()
