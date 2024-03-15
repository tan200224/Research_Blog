# Training a segmentation model

### 1. Specify the path of training(images and labels) and testing(images and labels) data. Zip each image and its label together
### 2. Transform using different methods
### 3. Use DataLoader to load the data and specify the number of batches
### 4. Train the model
### 5. Testing the model

#### 1. Data Processing
For data processing, I created 91 slices and 64 slices of Dicom files for both the images and the labels. 
Then, I converted them into Nifti files (now, each Nifti has the same volume, a group of 91 or 64). 
Finally, I checked the Nifti labels. to remove the folders where labels were empty(there was only 0 in the mask). 

### 2. Transform the data
There are a lot of ways to transform pancreatic CT scans. 
In my case, I have 159 files for groups of 90 and 257 for groups of 64. I used Monai to compose the data. 
The letter d at the end of each transformation means I'm doing it with a dictionary.
1. LoadImaged # It will load the data from the dictionary, which consists of ['img':training_images, 'seg':train_labels]
2. EnsureChannelFirstd # It adds a channel or dimension to our data, which is the number of batches. Therefore, the data will have a dimension that consists of [batch, width, high, volume]
3. Orientation # It will rotate the image in different ways
4. Spacingd # It will make sure that all of the data have the same size. It depends on how the CT scan was taken. The data might look different. Therefore, we want to make sure they are in the same size
5. ScaleIntensityRanged # It can change the contract in the image to make the organ more visible and then normalize the value to 0-1
6. CropForegroundd # It will crop the area is have nothing on it. It also has a source key, which should be set to be the training image instead of labels because I want to make sure we only crop the information from the image, not from the label
7. Resized # It is a optional process, however, if you used CropForegroundd, then you must resize the image becuase the crop is random, every image might have differnent size after crop. Therefore, you want to make sure they are the same size
8. ToTensor # It will convert the data into Tensor, which is needed to be used for training.

#### Load the data
Once I have the dictionary that contains images and labels and have the transform. I created my training data set with the dictionary and the transform. 
Then, I created a DataLoader with the Dataset and the number of batches. Since I dont have enough data at this moment I have to make the testing data same as my training data.
However, in the training, the testing data will not really affect the performance of the model. 

#### Training the data
As for training the data, I used UNet from Monai. and device as my local GPU: 4090.
As for the loss function, I used the DiceLoss from Monai.
As for the optimizer, I used Adam.

1. First, I started training my model with 600 epoches
2. As for training, I want to add up and print the metric and loss for each step (each image and label that is used to train).
3. Then I want to store the best metric and best loss for each epoch (Each time the model trained on the entire dataset).
4. The image and label will first be passed to the GPU: image, label = (batch_data['img'].to(device), batch_data['seg'].to(device))
5. Then, make sure the optimizer is set back to 0. Then get the output from the model(batch_data['img']).
6. Then calculate the loss by using the loss_function(predicted, target).
7. Once we have the loss function, I use it for backpropagation with the optimizer.
9. Then, calculate the metric by dice_metric(predicted, target)
10. I will add up all the losses when I am looping through each step
11. At the end of each epoch, we will take the average loss and dice metric
12. I also did the same thing for our testing data, just to create a loss and metric group for testing. the idea is very similar just stores the value of loss and metric


### Some Bug that I counter
1. 1. The output of the loss function is not Numpy, but tensor. When I store them in to a file. It gave an error for trying to create a spy file. It is fixed by just doing loss.item().
2. The value for spatial_size. When I was trying with the groups of 91. I try to do [128, 128, 91 or 90]. It caused an error. It only take 16, 32, 64, etc.

### Question
Which way is better, Jupyter Notebook or VScode?
Advantages of using .py and .ipynb?


## Result for groups of 90
<img width="1117" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/61ce81e6-f21d-4998-9571-eb624a431c88">

<img width="1117" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/dfe4c4bf-7ea0-499a-ba4b-4331a5b51e29">

<img width="1084" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1ae86f45-df97-48d1-b58a-4342fd093e2a">

<img width="1082" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/65fc41be-af6e-4d3a-b090-df67c7164a48">


## Result for group of 64
____________________

Epoch_loss: 0.0067

Epoch_metric:  0.9933

test_loss_epoch: 0.0560

test_dice_epoch: 0.9440

current epoch: 600 current mean dice: 0.9930

best mean dice: 0.9532 at epoch: 570

train completed, best metric: 0.9532 at epoch: 570

<img width="600" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/634ebd56-0955-4704-9a70-6e23f86af675">

<img width="491" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1a954b6a-0195-411b-a027-a3bf679f0f1f">

<img width="488" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/a065627e-39d8-4fbf-b5e9-de5d51fe3868">



