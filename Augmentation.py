import imageio
import imgaug.augmenters as iaa
import pandas as pd
import matplotlib
# import imgaug.random as iarandom
from numpy.random import Generator, PCG64
# %matplotlib inline

col_list = ["ImageId", "PredictionString"]
df = pd.read_csv('train.csv', usecols=col_list)
drop_images = ['ID_1a5a10365', 'ID_4d238ae90.jpg', 'ID_408f58e9f', 'ID_bb1d991f6', 'ID_c44983aeb'] 
df = df[~df['ImageId'].isin(drop_images)]
n = len(df)  # number of images in train dataset
train_aug = {}

for i in range(n):
    image_id = str(df['ImageId'][i]) + str('.jpg')
    prediction_string = df['PredictionString'][i]
    image = imageio.imread(image_id)
    # rng = iarandom.RNG(4)  # seed
    rng = Generator(PCG64())
    flag_flip = float(rng.integers(0, 2))  # flag for yaw *= -1, roll *= -1
    scale = rng.integers(800, 1200) / 1000  # ratio for change position (x,y,z)
    image = imageio.imread(image_id)
    seq = iaa.Sequential([
        iaa.Fliplr(p=flag_flip),
        iaa.Resize(scale),
        iaa.MultiplyHueAndSaturation((0.9, 1.1), per_channel=True),
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
        iaa.GammaContrast((0.5, 2.0))
    ])
    image_aug = seq(image=image)

    # save aug_image in same folder
    filename_aug = str('./') + image_id[:-4] + str('_aug') + image_id[-4:]
    imageio.imwrite(filename_aug, image_aug)
    filename_aug = image_id[:-4] + str('_aug') + image_id[-4:]
    # calculate prediction string if changed, type, yaw, pitch, roll, x, y, z
    prediction_string_aug = str(prediction_string).split()
    prediction_string_aug = [float(z) for z in prediction_string_aug]
    l = len(prediction_string_aug) // 7  # number of car in one image
    result = []  # prediction string for ith image
    for j in range(l):
        car_data = prediction_string_aug[7 * j:7 * (j + 1)]
        if flag_flip:
            car_data[0] *= -1
            car_data[2] *= -1
        car_data[3] *= scale
        car_data[4] *= scale
        car_data[5] *= scale
        for x in range(7):
            result.append(car_data[x])

    # create dict for csv file predictionstring
    if i == 0:
        train_aug['ImageId'] = []
        train_aug['PredictionString'] = []
    train_aug['ImageId'].append(filename_aug)
    train_aug['PredictionString'].append(result)
    print(n)
# write csv file
df_aug = pd.DataFrame(train_aug, columns=['ImageId', 'PredictionString'])
df_aug.to_csv(r'./train_aug.csv', index=False, header=True)