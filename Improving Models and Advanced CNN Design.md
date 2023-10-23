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
Unlike L1 and L2 
