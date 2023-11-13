# CNN History

<img width="300" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3993ead2-b65b-4280-b1d1-f95cd19c986e">


# LeNet
* LeNet was introduced in 1995 by LeCun
* It was developed for handwritten digit recognition for US zip codes

## LeNet Architecture
<img width="812" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/76418def-c0d8-4556-83fa-34c7153fa84e">


# AlexNet
* AlexNet was introduced in 2012 by Alex Krizevsky, llya Sutskever and Geoffery Hiton from University of Toronto.
* It was the ILSVRX winnter in 2012
* It contained eight layers with the first five being Convolutional Layers and the last 3 being FC layers.
* It has over 60 million parameters and was trained on two GPUs for over a week

## AlexNet Architecture

![image](https://github.com/tan200224/Blog/assets/68765056/8569724e-aacf-447a-9393-578dd814e280)


# VGGNet
* VGGNet was first introduced in 2014 by Oxford University Research Karen Simonyan and Andrew Zisserman.
* It achieved 92.7% top-5 Accuracy in ImageNet (1000 classes)
* VGG16 has 13 Conv Layers with 3 FC Layers
* VGG19 has 16 Conv Layer with 3 FC Layers

## VGG Architecture
<img width="428" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/a059b659-54d6-448d-9625-29e1272cb581">

**VGG is simple and reliable and provide good result. However, it usually takes a long time to train.**

# ResNets
* ResNets were introduced in 2015 by Microsoft Researchers 

Classical CNNS Give Worse Performance if too Deep

<img width="424" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/600176c1-7e6b-455a-ae91-b79e71d18cc5">

* In reality, a classical CNN architecture has lots of problems training if layers are very deep
* Often, performance gets worse when number layers are used. 

### Very Deep CNNs Experience Exploding or Vanishing Gradients
* In deep networks with N layers, N derivatives must be multiplied together to perform our gradient updates
* if a derivative is large, gradients increase exponentially or "explode."
* Likewise, if derivatives are small, they decrease exponentially or "vanish."

## ResNet of Residual Networks Help Solve This 

<img width="814" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/8efcb70b-4af7-4b76-a1af-b16f1501cb1f">


# Why Do ResNets Work?

<img width="865" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/bbbf7958-0993-4c60-ad75-2416ff7f852d">


# Mobile Net
* MobileNet was a CNN architecture developed by Google in 2017
* It was designed to be an efficient lightweight CNN that could be used in embedded devices and mobile phones
* Good CNNs are large (Weights are 100mb+)
* Relatively slow on inference (i.e. forward propagation)

## Making CNNs for Mobile/Embedded Devices
* Mobile or embedded systems typically have low computational power as they are made to be cheap and power-efficient
* Using CNNs on those devices required:
  * Training smaller models
  * Compressing existing models via methods such as pruning, distillation, or low-bit networks.
* MobileNet achieved the goal of being a good mobile phone model by using:
  * Depthwise Separable Convolutions
  * Two Hyper-parameters


<img width="860" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/995c23a1-ed20-443e-9183-375032695f16">

<img width="859" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ddc90557-de2c-46ec-a249-1f02e896ae63">


# Two Hyper Parameters
* MobileNets also features two hyper parameters that effectively reduce the size of the model
    * Width Multiplier - This thins the model at each layer
    * Resolution Multiplier - Reduces the input image size and thus reduces the internal representation of every subsequent layer. 

# Inception Network Motivation
* As you've seen there's a lot of parameter tweaking involved in CNNs
* Filter Size, stride, depth, padding, FC layers etc.
* Inception aimed to solve the filter size selection problem.

## Inception Network
* The inception v1 Network was introduced by Google in 2014
* It achieved state-of-the-art performance in the ImageNet (ILSVRC14) Challenge

<img width="794" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/d4001dda-85f8-43c4-9725-a782ac1e4bf6">

<img width="863" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/ad43deca-e324-4cd7-b68d-d64f8f513b25">

<img width="791" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/86c62665-5391-4dd7-a2f1-99a58c71c995">


# SqueezeNet
**A smaller CNN that maintains good accuracy**
* Introduced in 2016 by researchers at UCB and Stanford and DeepScale
* Aim was to make a highly accurate by small CNN. Why?
    * Less Communication across servers during distributed training
    * Smaller size allows it to be used on embedded systems like FPGAs
    * Faster to update models via the cloud
* It had 50x less parameters than AlexNet and performed 3x faster

## SqueezeNet Key Takeaway
* Replace 3x3 filters with of 1x1 - has 9x fewer parameters than a 3x3 filter
* Decrease the number of input channels to 3x3 filters
* Downsample later in the network so that convolution layers have larger activation maps 

<img width="773" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b7af7243-182d-44f4-af83-b257b82b279a">

# EfficientNet
* Introduced in 2019 by researchers at Google
* Motivation behind efficientNet
    * CNNs are typically designed at a fixed resource cost and then scaled up
    * Scaling works by either increasing depth or width
    * This is often tedious to experiment with, requires manual tuning and result in suboptimal result.
* What if there was a more principled method to scale up CNNs?

## EfficientNet's Scaling Method Compound Coefficient
* This method uniformly scales each dimension (width, depth, resolution) with a fixed set of scaling coefficients
* It utilizes Google's new AutoML
* The EfficientNet Family of models are able to surpass state-of-the-art accuracy with 10x better efficiency 


## Scaling
* Researchers systemically studied the effects of scaling up different dimensions.
* It was found that balancing scaling in all dimensions resulted in the best overall performance. 

<img width="827" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/f07ded18-36f6-468d-9939-505502e7c975">

## EfficientNet Architecture
* The compound scaling methodology can be applied to any CNN
* The effectiveness of model scaling depends heavily on the baseline network.
* Researchers developed a new baseline network by performing a neural architecture search using AutoML's MNAS framework, which optimizes both accuracy and efficiency
* Their new architecture using mobile inverted bottleneck convolution is similar to MobileNetV2, but is slightly larger due to the increased FLOP budgets

<img width="461" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/01f1a020-5b50-4686-a718-0a2f7d2d1ba3">


# DenseNEts
**ResNets but better!**
Densely Connected Convolutional Neural Networks
* Introduced in 2016 jointly by Cornwell University, Tsinghua University and Facebook AI Research
* It was able to attain higher accuracy than ResNet with fewer parameters


Motivation 
* Training Deep CNNs is problematic due to vanishing gradients
* This is because the path for the deep network becomes so long gradients go to zero before completing the path (vanish)
* DenseNets solve this by using an ingenious concept of "collective knowledge" where each layer received info from all previous layers

<img width="851" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/fe5a088c-ede9-49d1-8bf1-0001b763d4b2">

<img width="853" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/4dd59eaa-250a-4a4a-acb3-e471a2eb7e6c">

<img width="854" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/37e485b4-7f01-455d-ada3-eca5c6aeffe0">

<img width="862" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/3e6e3216-2637-42e9-ae9d-3abb397fe2e0">


# ImageNet
**Why ImageNet is so important to the computer vision World**
* Currently the world's largest dataset of labeled image

## ImageNet - ILSVR
* The most highly-used subset of ImageNet is the ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
* The dataset spans 1000 object classes and contains 1281167 training images, 50000 validation images and 100000 test images. This subset is available on Kaggle
* It is the most common benchmark used when evaluating new exotic CNN models. 

## What makes it so good? It's Size
<img width="595" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/247b3dd3-841c-47dc-bfab-6f215b7c9dc2">


# ImageNet- Current State of Art Performance 
<img width="858" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/185a24c1-5445-4cf7-b137-27d0ed107c5c">

<img width="869" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/a858a8f5-82f7-4e83-aaf6-6b12311dadf7">



