# Segmentation with Combined dataset and Synthetic data

## Data Processing
#### Real Data
1. For the real data, we wants to randomly shuffle and split into training and testing data
   1. To make sure the two dataset inside is combined dataset is seperated
   2. But we have to make sure that the label is also shuffled in the same way, we did it with zip
2. Load data from path, get fdata from nii, and change to float tensor
3. Apply cropping to get rid of black space
4. Clip the intensity in between -50 to 200, and normalize it to 0 and 1
5. We change the shape of the image into (depth, height, and width) for the segmentation model, and also add a channel number at the front (channel, depth, height, and width)
6. Lastly, we resize the image into (256, 256)


## Synthetic Data
### Data Prcoessing
1. Since the synthetic data was training with from the data that are transformed, so we only add changed the shape and adding a channel.

### Loading data
We used ConcatDataset to concat the real data and synthetic data together. The synthetic data and real data might have the same mask.

## Model
The model that we used is the 3D Unet

      model = UNet(
          spatial_dims = 3,
          in_channels = 1,
          out_channels = 2,
          channels = (16, 32, 64, 128, 256),
          strides = ((2, 2, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1)),
          num_res_units = 2,
          norm = Norm.BATCH
      ).to(device)

However, 4 slices are not able to have 5 layers. Therefore, we have to change the strides for each of the images to make sure that they are not reducing the depth to less than 1. 


## Result
### Model 128, alpha between -0.05 to 0.05
![Screenshot 2024-07-13 115320](https://github.com/user-attachments/assets/a829c471-87c3-41ba-91fc-6ab271c22ada)
![Screenshot 2024-07-13 115507](https://github.com/user-attachments/assets/e999ebe8-1bc6-4587-b8bc-08560692fd79)

![Screenshot 2024-07-13 115532](https://github.com/user-attachments/assets/0b3d2b28-ea3a-4013-b126-be7fa749d0bc)


### Model 64, alpha between -0.005 to 0.006
![Screenshot 2024-07-13 144507](https://github.com/user-attachments/assets/81062e0c-8f9d-4491-9993-e06e506c7d7e)
![Screenshot 2024-07-13 144734](https://github.com/user-attachments/assets/fea51d3a-0c9e-4bb8-97fc-ffcee47f5ce6)

![Screenshot 2024-07-13 144757](https://github.com/user-attachments/assets/6bbd1fe7-3a15-49bb-88ac-231c027db212)


