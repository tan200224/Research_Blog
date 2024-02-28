## Overfitting
* A very common issue that pagues model development is overfitting.
* It occurs when a model has "over-trained" on the training data and thus ends up performing poorly on our test of validation datasets.
* This often happens when we don't have enough data, have used too many features, or developed an overly complex model

## Generalization
* A measure of how well our model performs on new unseen data
* Model that overfits to the training data performs poorly on new data, hence having poor Generalization
* A model that stores a lot of information has the potential to be more accurate by leveraging more features, but that also puts it at the risk of storing irrelevant features.

<img width="787" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/cfb2616a-df9f-4f29-b017-2f93f584de6d">

<img width="869" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/1d5eaa72-bca4-4650-87ec-969b860da396">

## Avoiding Overfitting
* YUsing large datasets or reducing model complexity and ensuring the model is using the right features to classify objects
* Regularisation - is a method by which we control the model complexity and ensure the model is using the right features to classify objects.

<img width="665" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/53912bcb-c7b6-40f1-b4a7-11847f639355">

<img width="428" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/8046fbbc-e6f3-440f-b7fa-751f30d526ac">


# Regualarization
* Regularization is a technique used as an attempt to reduce overfitting.
* It allows us to control the model complexity and ensure the model is using the right feature to classify objects.

## Methods of Regularization 
* L1 & L2 regularization
* Data Augmentation
* Drop out
* Early Stopping
* Batch Normalization

# L1 and L2 Regularization
### Weight Constraining
* L1 and L2 regularization work by forcing parameters (weights and biases) to take small values.
* This works because by reducing the weight in our network, we decrease their effects on the activation function, so you don't have some filter overpowering other features when they get activated  

## L1 Regularization or L1-Norm or Lasso Regression

<img width="865" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/45290364-52f3-4a58-8bad-faacd4e5126a">

## L2 Regularization or Ridge Regression

<img width="855" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/99e0e0bb-2547-434d-9201-2d18802f3ea3">

## Differences between L1 and L2
* L1 weights shrink by a constant amount toward 0.
* By shrinking less important feature weights to zero, it acts like a feature selection algorithm, yielding sparse models.
* In L2, weights shrink by an amount proportional to the weights
* L2 penalizes large weights more than smaller weights less.
* L1 tends to concentrate the weight of the network to a relatively small number but high important connections. 


# Dropout
* Unlike L1 and L2 normalization, dropout doesn't modify the loss function
* It modifies the network itself instead
* It works by randomly dropping out a number of nodes of a layer during training

<img width="877" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/c9e83d8d-2e3b-4b9a-bd6b-b107e6547e43">

## Dropout Note
* The fraction of amount of nodes we drop depends on the Dropout Rate
* It forces the network to learn more robust/reliable features as it acts like we trained several different networks
* Effectively double the number of iterations required to converge
* In testing, we use all activations but reduce them by a factor of p to account for the missing activations during training. 

# Data Augmentation
* One of the most effective ways of reducing overfitting
* It solves the problem of not having enough data or not enough variation in our dataset
* Image datasets lend themselves easily to applying data augmentation

# Data Augmentation Variety 
Both Keras and PyTorch Provide data Augmentation functions that offer many image manipulations, such as:
* Flipping
* Brightness
* Rotation
* Zoom
* Cropping
* Skew

## More on Data Augmentation
* It is only done on our training dataset
* In most standard implementations, we aren't  creating new data, but we manipulating our existing input data


# Early Stopping
### The Over Training Problem
* At some point during our training process, our validation loss may stagnate or stop decreasing and sometimes actually start to increase
* At this point, you should not continue training
* The method by which we detect diminishing return in training and thus stop training our model is called early stop

<img width="874" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3c16a9cf-8fd8-4dca-8912-cff1629b4d5e">


# Batch Normalization
* The Batch Norm Technique helps coordinate the update of multiple layers in a model
* It standardizes the activations of the prior layer, thus scaling the output of the layer
* It reparamatizes the model to make some units always be standardized by definition
* It reduces internal covariate shifts.

## Implementing BatchNorm

<img width="862" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/5e2a53f6-4de5-4dd7-b9d0-243dec87f619">

## Advice on Using BatchNOrm in CNNS

<img width="826" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/09605cb9-1049-4b05-96e7-043390da3f51">


# When to Use Regularization
### Training without regularization first
* It is good practice always to train your model without regularization first
* Sometimes, regularization techniques can have detrimental effects on the model performance
* Dropout and Batch Norm also increase the convergence time
* It is always good to have a baseline model and introduce different regularization techniques one by one to assess its impact

# Regularization Warnings and Tips
* Dropout - Don't use it before the final softmax layer
* Regularization does not offer much help to smaller, less complex networks
* Not enough Epochs - With L2, Dropout, and data Augmentation, we do need more epochs to achieve the same performance
* Data Augmentation and Dropout add additional computations in training and can slow it down
* Under-fitting is possible if your L2 weight penalty is too high. 


















