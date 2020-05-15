import torch
import tqdm
import torch.optim as optim
import gc
import pandas as pd
from torch.optim import lr_scheduler
from load import load_data, train_data_test
PATH = 'Dataset/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criterion(output, mask, state):
    # output: result from nn;
    # mask: mask image given
    # state: state info for image points
    # existence
    pred = torch.sigmoid(output[:, 0]) # make probability of existence
    # binary classification loss
    exist_loss = -mask * torch.log(pred + 1e-10) - (1.0-mask) * torch.log(1.0 - pred + 1e-10)
    exist_loss = exist_loss.mean(0).sum()
    # state when existence
    pred_state = output[:, 1:] # states for each point
    # L1 loss
    state_loss = (torch.abs(pred_state - state).sum(1) * mask).sum(1).sum(1)/mask.sum(1).sum(1)
    state_loss = state_loss.mean(0)
    # total loss = existence + state
    loss = exist_loss + state_loss
    return loss


def train(epoch, history=None):
    model.train()

    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(tqdm(train_loader)):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        loss = criterion(output, mask_batch, regr_batch)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()

        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data))


def evaluate(epoch, history=None):
    model.eval()
    loss = 0

    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in validate_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss += criterion(output, mask_batch, regr_batch, size_average=False).data

    loss /= len(validate_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(loss))


data = train_data_test('train_small.csv')
train_loader, validate_loader = load_data(data)
epochs = 16
model = MyUNet(8).to(device) # model name
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(epochs, 10) * len(train_loader) // 3, gamma=0.1)

history = pd.DataFrame()

for epoch in range(epochs):
    torch.cuda.empty_cache()
    gc.collect()
    train(epoch, history)
    evaluate(epoch, history)

torch.save(model.state_dict(), './model.pth')