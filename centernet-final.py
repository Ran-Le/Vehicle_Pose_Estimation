from helper_functions import *

























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
