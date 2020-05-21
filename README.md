# CS231N_Project
## 1.Data
### 1) Create a folder in CS231N_Project called "Dataset"
### 2) Move all data downloaded from https://www.kaggle.com/c/pku-autonomous-driving/data to "Dataset"
### 3*) Currently try to create a small training dataset "train_small.csv" with fewer rows would be great --5/10/2020


## 2. Testing Logs

#### Update (05/18/20)
* Remove mesh from baseline: No difference observed
* <s>Test on **small dataset (400 iamges)**: Overfit</s>
* Baseline: **EfficientNet(b0)+UNet+CenterNet**, zero bg padding, single conv without batchnorm

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