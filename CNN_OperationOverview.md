# Building a CNN, Put all the pieces together and build a CNN

<img width="1613" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e6775261-b659-4186-8db2-9500cbb7c94c">

1. We have one input image to the CNN with a depth of 3
2. Then we applied 32 of the 26*26 filters on the image and got our result Conv_1
3. Then we applied the second ReLU with 64 layers of 24*24 filter and got our result Conv_2
4. We then applied the Conv_2 to Max Pool to produce a 12*12 feature map with 64 layers
5. Then we flatten the result of the Max pool by multiplying the height, width, and depth
6. Then we use the final connection to extract the data/information to connect to the output
7. finally, we apply Softmax to calculate the probabilities of each output. 

<img width="920" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/822e3775-3d99-4097-a699-6af611b58025">

<img width="919" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3e2debcc-ea47-4ed4-8a07-4e1a471834d9">

<img width="925" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/920be9e8-ec2f-4d4a-b8b8-ad88beb4615a">

<img width="928" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/86efde02-9ffa-4e05-ab0d-96f1761e47e8">

<img width="914" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/f27f12ab-2d90-4642-830d-a95022a720de">


# Parameter Counts in CNNS
### What are Parameters or weights
* They are the variables that need to be learnt when training a Model
* Often called learnable parameters or weights
* Our hidden layers, like the Convolution or Fully COnnected Layers have weights.


Calculating Learnable Parameters in a Conv Filter
<img width="1877" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/61f96424-3c16-4143-ad76-5722cd6d05c2">


### Biases
* Biases allow us to shift our activation function (left or right) by adding a constant value
* Biases are per 'neuron', so per filter in our case (shared in the case with color RGU images as the input)
* These shared biases per 'neuron' allow the filter to detect the same feature
* They are a learnable parameter


### Calculated the Number of Parameters in Our Simple CNN
<img width="881" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/c66c0df6-e157-4474-9c66-223403d4bb6e">


# Why CNNs Wiorks SO Well For Images

### Standard Neural Newtwork 
* Standard Neural Networks don't have Convolution Filter Inputs
* Layers with everything that is fully connected (28*28 image; the first layer will already have 784 input nodes)
* For the image, every pixel will be it is own input
* Not scalable to large data inputs
* Overfitting



### Advantages of Convolution Neural Networks
* Parameter sharing - where a single filter can be used all parts of an image
* Sparsity of connections - As we saw, fully connected layers in a typical Neural Network result in a weight matrix with a large number of parameters.
* Invariance - Max Pool Example




