# CS231N_Project

## 1.Setup
* Create a folder in CS231N_Project called "Dataset"
* Move all data downloaded from https://www.kaggle.com/c/pku-autonomous-driving/data to "Dataset"

## 2. Testing Logs

#### Update (05/10/20)
* Setup Google Cloud environment

#### Update (05/12/20)
* Setup utility functions

#### Update (05/15/20)
* Finish data loading

#### Update (05/16/20)
* Finish data preprocessing

#### Update (05/18/20)
* Remove mesh from baseline: No difference observed
* <s>Test on **small dataset (400 iamges)**: Overfit</s>
* Baseline: **EfficientNet(b0)+UNet+CenterNet**, zero bg padding, single conv without batchnorm
  * EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch
  * U-Net: https://arxiv.org/abs/1505.04597
  * CenterNet: https://arxiv.org/abs/1904.07850

#### Update (05/19/2020)
* <s>Replace **Batchnorm** with **Maxpooling+Upsamping**: Significant increase of loss</s>
* <s>Replace **Double conv** with **conv**: worse performance</s>
* Separate validation and testing set, prepare for cross validation
* Red bounding box for better visualization


#### Update (05/20/2020)
* <s>Remove **Batchnorm** from **Double conv**: slightly worse</s>
* Set random seed to 231 for easier comparison

#### Update (05/21/2020)
* Remove **left/right background zero padding**: No difference observed
* <s>Cross validation: negligible influence</s>
* Replace **Upsample** with **ConvTranspose2d**: Less overfit, similar performance 
* <ins>EfficientNet versions (b0-b5): better performance but slower training, try later</ins>
* Replace **ConvTranspose2d** with **ConvTranspose2d-BatchNorm-ReLU**: better performance

#### Update (05/21/2020)
* Separate detection map and pose estimation map: slightly worse

#### Update (05/23/2020)
* Submission: Private 0.034 / Public 0.033
* <s>L2 loss: nan error</s>

#### Update (05/24/2020)
* Add MultiRes Path: https://github.com/nibtehaz/MultiResUNet

#### Todo:
* change loss function
* Try stacked hourglass architecture: https://arxiv.org/abs/1603.06937
* Gaussian heatmap for CenterNet: https://arxiv.org/abs/1904.07850
* Upconv+RecurrentConv: https://arxiv.org/abs/1802.06955
* Dense U-Net: https://arxiv.org/abs/1808.10848