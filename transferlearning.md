# Transfer Learning
* In practice, very few people train an entire Convolutional Network from scratch
* Instead, it is common to pretrain a ConvNet on a very large dataset, and use the ConvNet as an initialization or a fixed feature extractor for the take of interest.

## Transfer Learning Rationale
* Training extensive models on vast image datasets like ImageNet can sometimes take weeks
* A model trained on so much data would have useful embeddings that can be applied to other image domains. Such as edge detectors, pattern and blob detectors
* Transfer learning is the concept where we utilize trained models in other domains to attain great accuracy as well as faster training times.

## Transfer Learning 2 Major Types
* Feature Extractor
* Fine Tuning

### Transfer Learning Feature Extractor 
**Freeze the CONV weights/layers - meaning they remain fixed**
**Replace the top layer with your top layer and train it**

<img width="737" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/f509470a-0ec7-46ee-94a2-f6195977908b">

### The Steps involved in Transfer Learning by Feature Extraction
1. Freeze the bottom layers of a pre-trained network
2. Replace the top half of the network with your top so that it outputs only the number of classes in your dataset
3. Train the model on your new dataset

## Transfer Learning - Fine Tunning

<img width="274" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/f953ac85-774c-4556-a8e7-9c8fe9a74d7e">

In Fine Tuning, we complete all the steps in Feature Extraction but then we:
1. Unfreeze all or parts of the pre-trained model
2. Train the model for a few epochs, this is where we "fine-tune" the weights of the  pre-trained model

The intuition behind this is earlier feature maps in ConvNet learn generic features, while the later layers learn specifics about the image dataset. By  fine tuning we change those specifics from the pre-trained model to the specifics of our datset. 

## When Do We Use Transfer Learning?
* Ideal - The new dataset is large and similar to the pre-trained original dataset. Models should not overfit.
* Not Ideal but recommended - New data is large but different
* if data is small, Transfer learning and fine-tuning can often overfit the training data. A useful idea at times is to train a linear classifier on the CNN outputs.


# Transfer Learning Advice
* * Learning Rates - use very small learning rate for pre-trained models, especially when fine-tuning. This is because we know the pre-trained weights are already very good and thus don't want to change time too much
* * Due to parameter sharing, you can train a pre-trained network on images of different sizes.
 
  










