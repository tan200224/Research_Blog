* The Vae that we built uses 2D convolution layers. We have experimented with 3D, but the computation power is too high, I was not able to run it on my rtx 4090

* Using slices of 4 tends to be better than slices of 64, in terms of the speed of the training and also the performance of reconstruction and generation 
