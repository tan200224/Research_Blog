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

#### 2. Transform the data
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

## Result for groups of 91
____________________
Epoch_loss: 0.0119

Epoch_metric:  0.9881

test_loss_epoch: 0.0282

test_dice_epoch: 0.9718

current epoch: 600 current mean dice: 0.9891

best mean dice: 0.9754 at epoch: 589

train completed, best metric: 0.9754 at epoch: 589

<img width="822" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/2de4612e-e586-4c0f-9919-ffb37decdf08">

<img width="490" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/c1dcbaad-8143-4a78-a533-d24effe41dac">

<img width="490" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/9acf59f4-2e0a-43cc-b902-1dd7127bab1c">


## Result for group of 64
____________________
Epoch_loss: 0.0067

Epoch_metric:  0.9933

test_loss_epoch: 0.0560

test_dice_epoch: 0.9440

current epoch: 600 current mean dice: 0.9930

best mean dice: 0.9532 at epoch: 570

train completed, best metric: 0.9532 at epoch: 570

<img width="917" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/4e7c62ed-8b57-46b6-a10f-21cbfe385064">

<img width="490" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/a065627e-39d8-4fbf-b5e9-de5d51fe3868">

<img width="490" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1a954b6a-0195-411b-a027-a3bf679f0f1f">

### Some Bug that I counter
1. The output of the loss function is not Numpy, but tensor. When I store them in to a file. It gave an error for trying to create a spy file. It is fixed by just doing loss.item().
2. The value for spatial_size. When I was trying with the groups of 91. I try to do [128, 128, 91 or 90]. It caused an error. It only take 16, 32, 64, etc.

### Question
1. Which way is better, Jupyter Notebook or VScode?
   - Personal preference.
2. What are the advantages of using .py and .ipynb?
   - Using .py, you have to compile it without any additional work, some cases where, you want to run your code on a supercomputer. You might want it to be a .py file because .ipynb is more like a script and more interactive. you need to run all the cells. However, .ipynb file allows you to save the running progress. It's not like .py when it crashes; you need to restart everything again. 
3. As I experiment and train the data, one finding I have is that the model is very sensitive to the change in the intensity. the model is tested on a dataset that is transformed with different intensities the result will be affected.
   - Usually, if we want to train or use our model to work on specific data, we need to transform them. Otherwise, it may not work. Therefore, it is okay, will the intensity is variant, we can transform those data. But it can be something to try on and see if it works well with different intensities. 
4. Is there a way to do data augmentation so the performance of the model will be invariant to the change in intensity, transform, rotation, etc?
   - There is
5. Is the more epochs, the better? When should I stop training?
    - This will depend on the training loss and testing loss. If it already converges, it is not too necessary to keep training it because it will get overfitted.
6. How should I change the learning rate?
    - The learning rate will also depend on how the test and training loss looks like. If it is decreasing at a linear rate, it means the value is too low. it might stuck at a local minimum. if it is too large, it might not be able to find a minimum.
7. What if we do edge detection on it?
    - The CNN behind the since will do edge detection. to find the feature of edges. 


### Additional
More data. As we looked over the dataset with 82 scans the first time, we were unable to find the labeled data. However, I was able to find them this time. However, they are kinda a little bit different from the dataset that we are currently using. I made a data viewer to look at the data. Here is a sample of data for each of the dataset.

<img width="723" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/3f8639ec-ae27-4159-b7e7-54d962325929">

<img width="735" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/e9576d2e-1a09-4a09-a5aa-ca9aea147195">



