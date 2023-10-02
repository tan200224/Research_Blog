# Loss Functions
### Quantifying Loss
* How bad are the probabilities we predict?
* How do we quantify the degree our prediction is off by?

Cross Entropy Loss
* Cross Entropy Loss uses two distributions, our ground truth distribution p(x) and q(x), our predicted distribution
* The lower the loss, the better the neural network is performing

<img width="405" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/67827ab4-4670-4b35-b427-804a4a4ac133">

## Other Loss Functions
* Loss Functions are sometimes called Cost Functions
* For the Binary Classification problem, we use Binary Cross-Entropy Loss (same as categorical cross-entropy loss, except it uses just one output node)
* For Regression, we often use the Mean Square Error (MSE)
* Other Loss Functions that are sometimes used:
  * L1, L2
  * Hinge Loss
  * Mean Absolute Error (MAE)

## What do we do with our quantified Loss
1. Updating all the weights of our model is not trivial
2. How do we correctly update our weights to minimize loss?
3. We use Back Propagation


# Back Propagation
### This is what makes Neural Networks Trainable
* The importance of Back Propagation cannot be understand
* Using the loss, it tells us how much to change/update the gradients so that we reduce the overall loss


<img width="696" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ca1d6d16-46fe-4e3e-a884-f7af61398531">

<img width="692" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/36646bf1-a324-4729-834f-cc5cd0d33b71">


Back Propagation Process 
* By forward propagating input data, we can back propagation to lower the weights to lower the loss.
* But, this simply tunes the weights for that particular input
* We improve Generalization
* By Continuously changing the weights for each data input (or a batch of images), we are lowering the overall loss for our training data.



### How does Back Propagation Work?
* Chain Rule!
  
<img width="838" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/06d5ecc8-735a-495c-a246-ebf3090e564d">

<img width="857" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ca7243ac-ebda-4643-8289-7bdb6e3f40f2">


# Gradient Descent
### Finding the optimal Weights


### Loss Functions How do we find the lowest loss?
* Back Propagation is the process we use to update the individual weights or  gradients
* Our goal is to find the right value of weights where the loss is the lowest
* The method by which we achieve this goal (i.e. updating all weights to lower the total loss) is called Gradient Descent)
* It's the point at which we find the optimal weights such that loss is near the lowest. 


## Gradients are the derivative of a function
* It tells us the rate of change of one variable with respect to the other
* A positive gradient means loss increases if weights increase
* A negative gradient means loss decreases if weights increase



## More on Gradients
* The point at which a Gradient is zero means that small changes to the left or right don't change the loss
* In training Neural Networks, this is good and bad
* Sometimes, this can be stuck in a local minima, but we want the best performance, which should be at the global minima


## Gradient Descent Method
* Naive Gradient Descent - Passes the entire dataset through our network then updates the weights.
* Stochastic Gradient Descent (SGD) - Updates weights after each data sample is forward propagated through our network
* Mini-Batch Gradient Descent - Combines both methods. It takes a batch of data points and forward propagates all, then updates the gradients.
    * This leads to faster training and convergence to the Global Minima
    * Batches are typically 8 to 256 size



# Optimisers and Learning Rate Schedules
### Methods or Algorithms used in finding optimal weights 

### The problems with standard SGD 
* Choosing an appropriate Learning Rate (LR), deciding Learning Rate Schedules
* Using the same learning rate for all parameter updates (as is the case with sparse data), but most importantly, SGD is susceptible to getting trapped in Local Minimas or Saddle Points
* To solve some of these issues, several other algorithms have been developed, including some extensions to SGD, which include Momentum and Nestor's Acceleration


## Momentum
* One of the issues with SGD is areas where our hyper-plane is much steeper in one direction
* This results in SGD oscillating around these slopes, making little progress to the minimum point.
* Momentum increases the strength of the updates for dimensions whose gradients switch direction. It also dampens oscillations. Typically we use a Momentum value of 0.9

<img width="401" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/d45c7e84-06ef-4b9d-834c-5830c6e572ee">


Nesterov's Acceleration
* One problem introduced by Momentum is overshooting the local minimum.
* Nestoerov's Acceleration is effectively a corrective update to the momentum, which lets us obtain an approximate idea of where our parameters will be after the update.

<img width="458" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/55a66535-d5e6-4d0e-a25d-a98949dee00b">



### Learning Rate Schedules 
Sometimes, you might not want to use an algorithm that uses advanced optimizers.  
* A preset list of learning rates used for each epoch.
* Progressively reduce over time
* We use LR Schedules because if our LR is too high, it can overshoot the minimum points
* Applying a progressively decreasing learning rate allows the network to take smaller steps (when gradients are updated), allowing our network to find the point of lowest loss instead of jumping over it.
* Early in the training process, we can afford to take big steps, however, as the decrease in loss slows, it is often better to use smaller learning rates to avoid oscillations
* Learning rate schedules are simply to implement with our deep learning libraries





