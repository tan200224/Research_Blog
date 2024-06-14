# VEA for Nifiti CT Scans 
The goal of this experiment with VAE is to be able to generate realistic synthetic 3D CT scans. The dataset that we have contains 440 groups of 64 slices of CT scans. 

# Method 
For this experiment, to approach the solution, we tried three different methods - in two of them, we used a 2D encoder and decoder, and in one of them, we used a 3D encoder and decoder. The different between a 2D and a 3D model is that. The 3D model will in theory work better on 3D images because it will consider more the correlation between the layer before and after the current image. 


## THe first method
![Screenshot 2024-06-04 161022](https://github.com/tan200224/Research_Blog/assets/68765056/3c560b4a-9da1-42e5-b59d-113587b47d58)

## The second method (2D)
The model (both encoder and decoder) contains a relatively small number of channels. We started the model with an input of 64 channels, but the channel numbers for the convolutional layers are very small, ranging from around 2 to 16. 
### The result of the model
The model can reconstruct some images. However, the reconstruction and generation work really poorly. The reconstruction is only able to generate a black-and-white image with a body shape. 

## The third method (2D)
This model contains a larger number of channels. We started the model with an input of 64 channels, but the channel numbers for the convolutional layers are very large, ranging from 16 to 512.
### The result of the model
The model can reconstruct the images better than the first method. The reconstruction provides more details of the original image; we can see some gray colors and some shapes of the organ in the CT scans. However, the generation is still working very poorly.
![Screenshot 2024-06-07 171238](https://github.com/tan200224/Research_Blog/assets/68765056/6581fa00-d7a2-4c9f-92ac-4f8fd9c56bca)

## The fourth method (3D)
For this approach, we are not able to get it to work yet. The model since to be more complicated and requires a lot of computation power. We were not even able to train the model due to the error that memory in the GPU was running out. The GPU that was used for this experiment is RTX 4090, which contains 24GB of RAM. 

# Challenage
1. The size of the model and input image can be confusing
2. The 3D model seem to take a lot of memory from the GPU

# Reference
[Training a Convolutional Variational Autoencoder on 3D CFD Turbulence Data](https://medium.com/@agrija9/training-a-convolutional-variational-autoencoder-on-3d-cfd-turbulence-data-7df8e207a58f)

[Tutorial: Abdominal CT Image Synthesis with Variational Autoencoders using PyTorch](https://medium.com/miccai-educational-initiative/tutorial-abdominal-ct-image-synthesis-with-variational-autoencoders-using-pytorch-933c29bb1c90)
