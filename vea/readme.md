* The Vae that we built uses 2D convolution layers. We have experimented with 3D, but the computation power is too high, I was not able to run it on my rtx 4090

* Using slices of 4 tends to be better than slices of 64, in terms of the speed of the training and also the performance of reconstruction and generation 


An updated of mask to image vae model

## Future steps
1. Train the model to generate synthetic medical images from the mask
2. Make some synthetic data and add it to the train set
3. Train a segmentation model to generate a mask from the image
4. classification model to predict if there is a tumor or not
5. Collecting results and see how much our method improved from the baseline 


## Mask to Image Generation

### Method 
We used the same model architecture as the VAE 2D; however, this time, instead of using the medical image to generate medical images, we decided to use a segmentation mask of the pancreas to generate medical images. Our goal is to hope that this will make the model focus more on the pancreas quality of the synthetic data. We also implemented a loss that focus on the pancreas by multiplying the segmentation with the medical image. 

### Result
![Screenshot 2024-06-27 133800](https://github.com/tan200224/Research_Blog/assets/68765056/dd681aa9-95ec-4491-bb75-35aabf3d68e1)

