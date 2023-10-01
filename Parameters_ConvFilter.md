# Kernel Size and Depth
### Sizing Convolution Filters 
Can we use other sizes?
Yes, but it has to be a square and an odd number because odd-sized filters are symmetrical around the center pixel or anchor point. The lack of symmetry here results in distortion across layers.

<img width="257" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/8bc9c78d-ab5a-4ccc-8ca1-6b8ecbfc76f0">

### Depth
* Typically refers to the color channels
* Depth can be used to refer to the 3rd dimension of any layer in our CNN




# Padding
How Conv Filters Produce an Output Sammler than the Input?

<img width="734" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ef44ffcc-5180-4aad-9afb-2b7d035736a8">

CNN can have several sequences of Convolution layers, which will decrease the size of the feature map as it progresses the network.

### Why use padding?
* For very deep networks, we don't want to keep reducing the size.
* Pixels at the edges contribute less to the output Feature Maps, thus we're throwing away information from them. (the pixels are the edges are only touched once when mapping the Kernel on it)

Therefore, we can use padding to prevent the feature maps from getting too small.
<img width="703" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/4277c3dd-9514-43ff-bdfe-b95e91c27672">




# Stride
### Stride is basically our step size, it defines how many steps we take when sliding out the Convolution Window across the input image.

### What does a Stride of 2 look like?

<img width="721" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b216aa17-7e63-4d85-bcea-ff33a086def2">

<img width="714" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e77530ef-8338-4863-8ba7-5d05723714e0">

<img width="731" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e0fda621-7342-4c7a-ad6e-7bda310b3912">

### Stride Observations
* A larger Stride produced a smaller Feature Map output
* Larger Stride has less overlap
* We can use stride to control the size of the Feature Map output

## Calculating Output Size
<img width="817" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/5efa7940-671a-4cef-85ea-2bb90342cd6c">

<img width="809" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/c728f4bc-8c75-4e80-8800-c67766fa857c">

**NOTE**: If we get an integer value for our output size, we find it down to the nearest integer




# Activation Functions

### The purpose of Activation Functions 

To enable the learning of complex patterns in our data. 
* Introduces non-linearity to our network
* This allows a non-linear decision boundary via non-linear combinations of the weight and inputs


Rectified Linear Units (ReLU) have become the activation function of choice for CNNS.

ReLU is advantageous in CNN Training:
* Simple Computation (Fast to train)
* Does not saturate

<img width="347" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/d2d9397c-1756-4e33-b0ac-11ac1dbaea21">

### The ReLu Operation
* Change all negative values to 0
* Leave all positive value alone

## Applying the ReLU Activation 
<img width="817" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/4512905a-9ae6-44c6-afa5-e10f9fcfe531">


### Exampple of a Rectified Linear Map

<img width="664" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e8966dd3-456c-41ee-abce-ff6b41ceb8b5">

All the negative values are changed to 0, and all the white values remain the same.




# Pooling
* Pooling is the process whereby we reduce the size or dimensionality of the feature map.
* This allows us to reduce the number of Parameters in our Network whilst retaining important features.
* Also called subsampling or downsampling

Example of Max Pooling

<img width="593" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/f9ad99dd-2a75-494d-9776-e1da968c2eec">

As the image shown, the Max Pooling Operation simply took the max value from the 2*2 block in the corresponding colors
  
### More on Pooling
* Typically, we use 2*2 kernels and a Stride with no padding.
* With the above setting, pooling reduces the dimensionality by a factor of 2.
* Pooling makes our model more invariant to minor transformations and distortions in our image.
* We can also use Average Pooling or Sum Pooling.


### Why Pooling Works?
### Pooling reduces our feature map size by half in most cases. Is that OK?
* Neighbouring pixels are strongly correlated, especially in the lowest layers.
* Remember, the further apart two pixels are from each other, the less correlated.
* Therefore, we can reduce the size of the output by pooling the filter response without losing information
* A big stride in the pooling layer leads to high information loss.
* In practice, a stride of 2 and a kernel size 2*2 for the pooling layer were found to be effective in practice. 















