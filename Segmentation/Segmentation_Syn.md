# Segmentation with Combined dataset and Synthetic data

## Data Processing
#### Real Data
1. For the real data, we wants to randomly shuffle and split into training and testing data
   1. To make sure the two dataset inside is combined dataset is seperated
   2. But we have to make sure that the label is also shuffle in the same way, we did it with zip
2. Load data from path, get fdata from nii, and change to float tensor
3. Apply cropping to get rid of black space
4. Clip the intensity in between -50 to 200, and normalize it to 0 and 1
5. 
#### Synthetic Data
