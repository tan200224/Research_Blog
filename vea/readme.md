* The Vae that we built uses 2D convolution layers. We have experimented with 3D, but the computation power is too high, I was not able to run it on my rtx 4090

* Using slices of 4 tends to be better than slices of 64, in terms of the speed of the training and also the performance of reconstruction and generation 


An updated of mask to image vae model

## Future steps
1. Train the model to generate synthetic medical images from the mask
2. Make some synthetic data and add it to the train set
3. Train a segmentation model to generate a mask from the image
4. classification model to predict if there is a tumor or not
5. Collecting result and see how much our method improved from the baseline 
