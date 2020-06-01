import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2

from efficientnet_pytorch import EfficientNet

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

effnet_ver = 'b0'
dropout_rate = 0.3


# dropout_rate = 0.0

# def set_dropout(model, drop_rate):
#     # source: https://discuss.pytorch.org/t/how-to-increase-dropout-rate-during-training/58107/4
#     for name, child in model.named_children():
#         if isinstance(child, torch.nn.Dropout):
#             child.p = drop_rate
#             print("name:", name)
#             print("children:\n", child)


# def effnet_dropout(drop_rate):
#     base_model0 = EfficientNet.from_pretrained(f"efficientnet-{effnet_ver}")
#     set_dropout(base_model0, drop_rate)
#     return base_model0


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
            nn.Conv2d(in_ch, out_ch // 8, 3, padding=1),
            nn.BatchNorm2d(out_ch // 8),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch // 8, out_ch // 8, 3, padding=1),
            nn.BatchNorm2d(out_ch // 8),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_ch // 8, out_ch // 4, 3, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            nn.ReLU(inplace=True)
        )
        self.bn = nn.BatchNorm2d(out_ch // 2)
        self.sc = nn.Conv2d(in_ch, out_ch // 2, 1)
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
            nn.Conv2d(in_ch, out_ch // 2, 3, padding=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 1),
            nn.BatchNorm2d(out_ch // 2)
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


class up(nn.Module):
    '''(Respath+ConvT)=>ResBlock '''
    '''in_ch1(ConvT),in_ch2(Respath)=>out_ch,dim_out==2*dim_in '''

    def __init__(self, in_ch1, in_ch2, out_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch1, in_ch1, 2, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(in_ch1)
        self.relu = nn.ReLU()

        self.respath = res_path(in_ch2, 2 * in_ch2)

        self.conv = double_conv(in_ch1 + 2 * in_ch2, out_ch)

    #         self.conv = res_block(in_ch1+2*in_ch2, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        x2 = self.respath(x2)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
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

        if effnet_ver == 'b0':
            self.up1 = up(1280, 1024, 512)
        elif effnet_ver == 'b1':
            self.up1 = up(1280, 1024, 512)
        elif effnet_ver == 'b2':
            self.up1 = up(1408, 1024, 512)
        elif effnet_ver == 'b3':
            self.up1 = up(1536, 1024, 512)
        elif effnet_ver == 'b4':
            self.up1 = up(1792, 1024, 512)
        elif effnet_ver == 'b5':
            self.up1 = up(2048, 1024, 512)
        #         self.up1 = up(1536,1024, 512)
        self.up2 = up(512, 512, 256)
        #         self.outc = nn.Conv2d(256, n_classes, 1)
        self.poseconv = output_conv(256, 1024, 7)
        self.detectionconv = output_conv(256, 256, 1)

    def forward(self, x):
        print("1:", x)
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

        #         x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        #         # torch.Size([1, 3, 320, 768])
        #         feats = self.base_model.extract_features(x_center)

        # # torch.Size([1, 1280, 10, 24])

        #         bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        #         feats = torch.cat([bg, feats, bg], 3)
        # # torch.Size([1, 1280, 10, 30])
        feats = self.base_model.extract_features(x)
        print('feat:', feats)
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



