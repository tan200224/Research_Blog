# What do CNN Learn?
* Learning involves adjusting weights/parameters during training that lead to the lowest loss

## How do filters work?
* The convolution operation is the dot product of the input vector and weight vector
* The dot product between two vectors is proportional to the angle between vectors
* The output is high when the angle between vectors is 0 degrees (vectors are in the same direction)

## Implementation
* Obtain the weights and bias of a filter
* Normalize weights between 0 and 1
* Use Matplotlib to plot the weight values in 2D


# Filter Activation of Convolution Neural Network
## Filter Activations
* When an input image is fed into our CNN, filters (some) "activate."
* This is the output of the ReLU layer

<img width="649" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/9269f98f-3bca-416f-b2bb-480bef5aa51c">

We can visualize the area of the image that corresponds to the filter or the actual activation as they go true.

## Visualizing Filter Activations

<img width="446" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/68fe65f0-fcca-4154-bbc1-9102a5a53ffa">

Each small image represents the output of the filter
* The outputs tend to appear sparse and localized
* Often, we can spot dead filters

## Another Example

<img width="565" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/6d2814ee-78b1-44eb-8788-c26a7a4e83a1">



<img width="1894" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/59432264-8946-4a76-8980-fff1ae41c73f">
* We can see as we progress from the early layers (left) to the deeper layers (right) our feature maps show less detail
* This shows that the "image" dimensions get smaller as they progress through the CNN (most cases) and that the deeper Conv layers are activated by combinations of lower layers
* That enables them to learn more complex patterns (a combo of lower filters is needed to activate an upper layer)

<img width="1061" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/9511e186-4cc3-469b-88eb-34db1c41fcba">

## Implementation
* Create a new model that takes an image as the input and outputs only the feature map
* Load our image and normalize it
* Propagate our input to our new model
* Extract the feature map response we want and use matplotlib to visualize it

# Maximizing Filters
## What inputs maximize our filter?
<img width="629" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e9c903ef-616b-4add-8da2-02e48c0dbb02">


## What inputs maximize our filter?
<img width="699" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/68a77cb5-f0d2-4ec9-8d9a-17117c724ef6">

## Understanding Maximizing Filters
* Viewing the input that maximizes each filter gives us a nice visualization of the CNN's modular-hierarchical decomposition of its visual space
* The first layer basically just encodes direction and color. These direction and color filters combine into basic grid and spot textures. These textures gradually get combined into increasingly complex patterns. 

<img width="716" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/a031713a-c8fb-4837-a3b7-4b4040aa4438">


## Implementation
* Load the trained model
* Define a loss function that seeks to maximize the activation of a specific filter in a specific layer
* We normalize the gradient of the pixels of the input image, which avoids very small and very large gradients and ensures a smooth gradient ascent process


# Maximizing Class Activations
**What input images maximize their class representations?**

## Class Maximization
* Let's say we trained a CNN to classify cats vs dogs
* But does our CNN know what a cat actually looks like?
* What input makes our CNN output at 100% certainty that it's seeing a cat?

<img width="847" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/932f3935-abd8-4f3a-9861-276f2ad7fdfd">

## Class Maximization Takeaways
* CNN's internalizes local features that bear resemblance to the class it's trained to recognize
* This shows that CNNs don't learn like we do. They understand a decomposition of the visual input space as a hierarchical-modular network of convolution filters
* The internal network is then a probabilistic mapping between combinations of these filters and their class labels. 

<img width="899" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/2e8e49d4-b1ab-4704-9a57-2bb58776c9bc">

## One Pixel Attacks
<img width="266" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b8e32762-2357-4944-a393-bdeddc31210d">

# Grad-CAM: Visualize What Influences Your Model

<img width="845" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3b056367-fe7c-473e-86c5-7c3c25013246">

<img width="1106" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/80d10a69-0201-4569-a8f9-65c8aca4ea04">

## How does Grad-CAM work?
* Grad-CAM exploits the spatial information that is preserved in Conv Layers
* It uses the feature maps produced by the last CNN layer
* We insert some differentiable (so that we can get the gradients) layers after the last Conv filter outputs
* We weigh the feature maps using "alpha values" that we calculated based on gradients
* This is used to make a heat map, we can create one for each output class



















