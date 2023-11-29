# What Are GANs?
First introduced in 2014 by lan Goodfellow et al. GANs are a type of Neural Network that generates data that plausible comes fomr an existing dirbution of samples.

### Examples of GANs

### Realistic Photo Gerneration

<img width="565" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/c240d6d5-306b-4744-8b75-397b977f5a70">

### Image-to-Image Translation

<img width="556" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/7a935e1d-a4e0-48a0-828a-5774826065d7">

### Text-to-Image Translations

<img width="497" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/71fcb27f-b934-4fd5-a9a6-3448fe28d7eb">

### Semantic-Image-to-Photo Translation

<img width="728" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/0c27a28c-2d52-4f59-8f08-52bff752a186">

### Super Resolution (SRGAN)

<img width="678" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/8c370a87-1430-4d26-9ad2-2b4dc5268083">


# Training GANs
* Training GANs is notoriously difficult compared to Neural Networks we use gradient descent to change our weights to reduce our loss.
* In a GANs, every weight change can change the entire balance of our dynamic system.
* We are not seeking to minimize loss, but finding an equilibrium between our two opposing Networks
* Training stops when the Discriminator can not tell apart Real vs Fake Data

## The Training Process
1. We randomly generate a noisy vector
2. input this into our Generator Network to generate sample data
3. We then take some sample data from our real data and mix it with some of our generated data
4. Train our discriminator to classify this mixed dataset and thus update its weight accordingly
5. Now we train the generator, make more random noisy vectors, and create synthetic data. With the weights of the discriminator frozen, we use the feedback from the discriminator to update the weights of the generator. 

## Are There Any Issues With GANs?

<img width="532" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b304645a-8de6-48c2-aaeb-e6d12075e951">

## Challanges in training 
* Achieving Equilibrium
* Time
* Bad Initializations - Caseing the discriminator loss to go close to zero
* Mode Collapse - This happens when, regardless of the noise input fed into your generator, the generated output varies very little. It occurs when a small set of images looks good to the discriminator and get scored better than other images. The GAN simply learns to reproduce those images over and over.


# Use Case of GANs

<img width="569" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/a06c3fb9-50ae-431a-ae94-efb2f4fff3fc">




