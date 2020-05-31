import torch
from tqdm import tqdm
import torch.optim as optim
import gc
import pandas as pd
from torch.optim import lr_scheduler
from load import load_data, train_data_test, camera
from model import MyUNet
import matplotlib.pyplot as plt
import numpy as np
from util import get_coords, str2coords
from sklearn.linear_model import LinearRegression
from visualize import plt_cars_coords
import cv2
from visualize import plt_cars
PATH = 'Dataset/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criterion(output, mask, state, size_average=True):
    # output: result from nn;
    # mask: mask image given
    # state: state info for image points
    # existence
    pred = torch.sigmoid(output[:, 0]) # make probability of existence
    # binary classification loss
    exist_loss = -mask * torch.log(pred + 1e-12) - (1.0-mask) * torch.log(1.0 - pred + 1e-12)
    exist_loss = exist_loss.mean(0).sum()
    # state when existence
    pred_state = output[:, 1:] # states for each point
    # L1 loss
    state_loss = (torch.abs(pred_state - state).sum(1) * mask).sum(1).sum(1)/mask.sum(1).sum(1)
    state_loss = state_loss.mean(0)
    # total loss = existence + state
    gamma = 5.0
    loss = exist_loss + gamma * state_loss
    if not size_average:
        loss *= output.shape[0]
    return exist_loss, state_loss, loss


def train(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        exist_loss, state_loss, loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_exist_loss'] = exist_loss.data.cpu().numpy()
            history.loc[epoch + batch_idx / len(train_loader), 'train_state_loss'] = state_loss.data.cpu().numpy()
        loss.backward()

        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))
    print('Train exist loss: {:.4f}'.format(exist_loss))
    print('Train state loss: {:.4f}'.format(state_loss))

def evaluate(epoch, history=None):
    model.eval()
    loss = 0
    exist_loss = 0
    state_loss = 0

    with torch.no_grad():
        for img_batch, exist_batch, state_batch in validate_loader:
            img_batch = img_batch.to(device)
            exist_batch = exist_batch.to(device)
            state_batch = state_batch.to(device)

            output = model(img_batch)
            exist_loss_t, state_loss_t, loss_t = criterion(output, exist_batch, state_batch, False)
            exist_loss += exist_loss_t
            state_loss += state_loss_t
            loss += loss_t

    loss /= len(validate_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'dev_mask_loss'] = exist_loss.cpu().numpy()
        history.loc[epoch, 'dev_regr_loss'] = state_loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))
    print('Dev exist loss: {:.4f}'.format(exist_loss))
    print('Dev state loss: {:.4f}'.format(state_loss))

def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

if __name__ == "__main__":

    cameraMat = camera()

    data = train_data_test('train.csv')
    train_loader, validate_loader, validate_data, validate = load_data(data)
    epochs = 2
    model = MyUNet(8).to(device) # model name
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(epochs, 10) * len(train_loader) // 3, gamma=0.1)

    history = pd.DataFrame()

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train(epoch, history)
        evaluate(epoch, history)

    # torch.save(model.state_dict(), './model.pth')
    history['train_loss'].iloc[100:].plot()
    plt.title('Training Loss')

    points_df = pd.DataFrame()
    for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
        arr = []
        for ps in data['PredictionString']:
            coords = str2coords(ps)
            arr += [c[col] for c in coords]
        points_df[col] = arr

    slope = LinearRegression()
    X = points_df[['x', 'z']]
    y = points_df['y']
    slope.fit(X, y)

    train_images_dir = PATH + 'train_images/{}.jpg'
    gc.collect()
    for idx in range(8):
        img, mask, regr = validate_data[idx]
        #     img, mask, regr = test_dataset[idx]

        output = model(torch.tensor(img[None]).to(device)).data.cpu().numpy()
        coords_pred = get_coords(output[0], slope, threshold=-0.5)
        coords_true = get_coords(np.concatenate([mask[None], regr], 0), slope)

        img = imread(train_images_dir.format(validate['ImageId'].iloc[idx]))
        #     img = imread(train_images_dir.format(df_test['ImageId'].iloc[idx]))

        fig, axes = plt.subplots(1, 2, figsize=(30, 30))
        axes[0].set_title('Ground truth')
        axes[0].imshow(plt_cars_coords(img, cameraMat, coords_true))
        axes[1].set_title('Prediction')
        axes[1].imshow(plt_cars_coords(img, cameraMat, coords_pred))
        plt.show()
