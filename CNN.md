# Convolutional Neural Network
Convolutional Neural Network uses filters to extract features from different parts of the image. 

<img width="1096" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/87613faf-0a31-4f92-8e63-95d4dfd430ff">

The Convolution Operation
* Mathematical term to describe the process of combining two functions to produce a third
* In our situation, the output is called a Feature Map
* We use a matrix, called a Filter or Kernel, that is applied to our Image
* So the first 'Function' is the image that is combined with the kernel or Filter, which produces a Feature Map

Image * Kernal = Feature Map

### Example of a Convolution Operation
<img width="975" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/17cfb763-8ad9-4270-939d-42e532eaccb2">


## Image Features
* Our feature maps are actually Feature Detectors
* Why did we do this?
* Because Convolution Filters or Kernels detect features in images

### Example of Convolution Filter as an Edge Detector
<img width="958" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/99f42874-1b97-4f6d-8cfb-381163b294f5">


## Calculating Feature Map Size
Feature Map Size = n - f + 1 = m
<img width="965" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/4689f76b-77ba-4783-9029-f9bf409770c4">

**NOTE** All images in CNN are square images


# Feature Detector
<img width="1133" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/1cd0cbdf-aba3-4b28-93ff-bd67a4006765">

In the past, a lot of features were hand-crafted, which was very hard, messy and often led to poor results. CNN solved this by having the ability to learn Features.


# Convolution Operations on Color Images
<img width="1073" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/34cab874-0b3d-4084-b7ef-243ec11a3f4b">
### The Advantages of Having a Filter For Each Colour:
* We can detect features that are specific to a Colour

### How do Multiple Filters Affect Our Output?
<img width="973" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/74e729c9-1fb2-4e03-97bf-e920b9fd1df0">






