# CS231N_Project
## 1.Data
### 1) Create a folder in CS231N_Project called "Dataset"
### 2) Move all data downloaded from https://www.kaggle.com/c/pku-autonomous-driving/data to "Dataset"
### 3*) Currently try to create a small training dataset "train_small.csv" with fewer rows would be great --5/10/2020


## 2. Testing Logs

#### Update (05/18/20)
* Remove mesh from baseline: No difference observed
* <s>Test on **small dataset** (400 iamges): Clear overfit</s>

#### Update (05/19/2020)
* <s>Replace **Batchnorm** with **Maxpooling+Upsamping**: Significant increase of loss</s>
* <s>Replace **Double conv** with **conv**: worse performance</s>
* Baseline: single conv

#### Update (05/20/2020)
* <s>Remove **Batchnorm** from **Double conv**: slightly worse</s>

#### Update (05/21/2020)
* Remove **left/right background zero padding**: No difference observed