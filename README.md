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
* Remove mesh from baseline: No observable difference
* <s>Test on **small dataset (400 iamges)**: Overfit</s>
* Baseline: **EfficientNet(b0)+UNet+CenterNet**, bg padding, single conv
  * EfficientNet: https://github.com/lukemelas/EfficientNet-PyTorch
  * U-Net: https://arxiv.org/abs/1505.04597
  * CenterNet: https://arxiv.org/abs/1904.07850

#### Update (05/19/2020)
* <s>Replace **Batchnorm** with **Maxpooling+Upsamping**: Significant increase of loss</s>
* <s>Replace **Double conv** with **Conv**: worse performance</s>
* <s>Separate validation and testing set, prepare for cross validation</s>
* Red bounding box for better visualization


#### Update (05/20/2020)
* <s>Remove **Batchnorm** from **Double conv**: slightly worse</s>
* Set random seed to 231 for easier comparison

#### Update (05/21/2020)
* Remove **background zero padding**: No observable difference
* <s>Cross validation: negligible influence</s>
* Replace **Upsample** with **ConvTranspose2d**: Less overfit, similar performance 
* Easy transfer between EfficientNet versions (b0-b5): better performance but slower training
* Replace **ConvTranspose2d** with **ConvTranspose2d-BatchNorm-ReLU**: better performance

#### Update (05/22/2020)
* Separate detection map and pose estimation map for implementation of CenterNet: slightly worse

#### Update (05/23/2020)
* Setup submission mechanisms for model evaluation
* Submission: Private 0.034 / Public 0.033
* <s>Replace L1 loss with L2 loss: nan error</s>

#### Update (05/24/2020)
* Implement MultiRes architecture: https://github.com/nibtehaz/MultiResUNet
* Submission: Private 0.025 / Public 0.029
* MultiRes Path: fix bugs, add ResPath shortcuts, increase stack number to 4, add BN layers, add epoch #, decrease learning rate, use **EfficientNet-B3**: significant overfit
* Submission: Private 0.031 / Public 0.031

#### Update (05/25/2020)
* Reset learning hyperparameters, add L2 regularization, weight_decay=0.01
* Submission: Private 0.035 / Public 0.031
* Replace **Double conv** with **MultiRes**
* Simplify **ResPath**
* Remove L2 regularization
* Clean up input data: Remove invalid images, scale up input resolution by 1.5
* decrease batch size to 2 for memory issue
* Base model dropout rate: 0.02
* Prepare for augmentation