# Dice Coefficient
we often hear about the Dice coefficient, and sometimes we see the term dice loss. A lot of us get confused between these two metrics. **Physically, they are the same, but when we look at their values, we find that they are not the same!**

# What is Dice Coefficient
When we want to produce an accurate model, we want to evaluate the model either during the validation steps or the testing step. We always need to calculate a metric, which is an equation between the ground truth and the predicted mask. And by looking at the value of those metrics we can say that the model is learning well or not. 

The dice coefficient, which can be used as a metric, is two times the intersection between the ground truth and the predicted mask, divided by the sum of the ground truth and the predicted mask. 

<img width="658" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/ea0b10bb-85b3-4a27-bf6d-70fc93138bdd">

As we can see the more intersection between A and B, the value of the dice value will go up. 

If there is no intersection the union of A and B will be 0, which means there is no intersection between the predicted mask and the ground truth. This will return a 0.

# The Dice Loss
<img width="203" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/9c3b8aa2-376e-4d33-90b5-c35c75436bd0">

# Question
What is ground truth, and what do metrics look like in training? 

# Reference
[PycadBlog](https://pycad.co/the-difference-between-dice-and-dice-loss/)
