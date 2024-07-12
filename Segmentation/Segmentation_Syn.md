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
          strides = ((2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
          num_res_units = 2,
          norm = Norm.BATCH
      ).to(device)

However, 4 slices are not able to have 5 layers. Therefore, we have to change the strides for each of the images to make sure that they are not reducing the depth to less than 1. 
