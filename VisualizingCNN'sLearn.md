# What do CNN's Learn?
* Learning involes adjusting weights/parameters during training that lead to the lowest loss

## How do filter work?
* The convolution operation is the dot product of the input vector and weight vector
* The dot product between two vectors is proportional to the angle between vectors
* The output is high when the angle between vector is 0 degree (vectors are in the same direction)

## Implementation
* Obtain the weights and bias of a filter
* Normalize weights between 0 and 1
* Use Matplotlib to plot the weight values in 2D


# Filter Activation of Convolution Netrual Network
## Fukter Activations
* WHen an input image is fed into our CNN, filters (some) "activate"
* This is the output of the ReLU layer

<img width="649" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/9269f98f-3bca-416f-b2bb-480bef5aa51c">

We can vistualize the area of the image thta corresponding the filter, or the actual activation as they go true

## Visualizing Filter Activations

<img width="446" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/68fe65f0-fcca-4154-bbc1-9102a5a53ffa">

Each small images represent the output of the filter
* The outputs tend apprear sparse and localized
* Often we can spot dead fitlers

## Another Example

<img width="565" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/6d2814ee-78b1-44eb-8788-c26a7a4e83a1">



<img width="1894" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/59432264-8946-4a76-8980-fff1ae41c73f">
* We can see as we progress from the early layers (left) to the deeper layers (right) our feature maps shows less detail
* This shows that the "image" dimensions get smaller as the progress throught the CNN (most cases) and that the deeper Conv layers are activated by combinations of lower layers
* That enables them to learn more complex patterns (a combo of lower filters are needed to activate an upper layer)

<img width="1061" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/9511e186-4cc3-469b-88eb-34db1c41fcba">

## Implementation
* Create a new model that takes an image as the input and outputs only the feature map
* Load our image and normalize it
* Propagate our input to our new model
* Extract the feature map response we want and use maplotilb to visualize it


