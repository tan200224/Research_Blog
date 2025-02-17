# Improving Early Diagnosis of Pancreatic Cancer with generative AI models
Zhuohao(Archie) Tan and Dr. Scott Spurlock

## Problem/Background: 
Pancreatic cancer is a disease with a high mortality rate due to a lack of early symptoms and data scarcity. Recent research has focused on improving AI models to produce more accurate diagnoses given medical images such as CT scans. Existing AI diagnosis models, trained on publicly available medical image datasets, often struggle with diagnosis because of limited training data. Our research provides an innovative approach to improving training data, which can be used to make existing diagnosis models better. We seek to train a model to generate realistic synthetic CT scans, which can then be used to improve the performance of pancreas segmentation and classification models. Our research investigates multiple generative models, including GAN (Generative Adversarial Network) and VAE (Variational Autoencoder). After training the generative models with real CT scans, the models generate synthetic CT scan datasets that mimic realistic pancreatic cancer CT scans; these realistic CT scans contain small tumors that doctors easily miss. To evaluate the performance of different synthetic datasets, multiple model training experiments combine various ratios of synthetic and real data. Two common metrics, Dice value and Binary Cross Entropy (BCE) are used to understand how realistic the synthetic images are, as well as their correlation with real pancreas annotations. Overall, our anticipated result is to generate realistic CT scans that can help train more accurate pancreatic cancer diagnosis models in the future.

## Datasets:
![image](https://github.com/user-attachments/assets/baa3e03a-6d73-4de1-81de-2a84abe860ce)


## Generate CT Scan Past Attempts 
In the past, we have attempted different approaches and built different models to generate synthetic CT Scan:
By using VAE, we want to be able to have some control over adjusting how the synthetic CT scan looks like 
1. 3D VAE model
     * The model is more complicated
     * Requires a lot of computation power
2. 2D VAE model
     * Easy to implement and build upon
     * The result is not bad
3. Mask to CT scan model
     * The VAE encodes the mask/pancreas annotation and decodes it back to CT Scan 
4. VAE-GAN model
     * The VAE encodes the mask/pancreas annotation and decodes it back to CT Scan
     * Added a discriminator to lower the blurriness of the CT scan image quality

![image](https://github.com/user-attachments/assets/1581efe0-89c8-4c12-a4ba-3b17ebb47689)
![image](https://github.com/user-attachments/assets/80547cb7-40da-40f2-95bb-742181fe7ae2)

# Synthetic CT Scan Image View
The smaller 𝞼 value indicates the less difference from the original image.

The overall result shows that the quality of this synthetic CT scan still needs to be improved, reducing blurriness

![image](https://github.com/user-attachments/assets/cd01df4a-9c40-468b-b69a-6f0ed06e482c)

# Segmentation Model
The segmentation model is used to locate where the pancreas is from the CT Scan

![image](https://github.com/user-attachments/assets/e8fdf2de-d5a3-4b5f-8caf-bf7cde4cd8fe)

### Evaluate the Segmentation Model Using Dice Value
The dice value measures the intersection area between the doctor's annotation and the model's prediction of the pancreas' location 

![image](https://github.com/user-attachments/assets/22eb41e4-6822-47e3-a16e-7b16e9148eeb)

### Training the segmentation model:

![image](https://github.com/user-attachments/assets/1c9b7b86-6d25-4c5c-ad64-e55a92fafc98)

# Current Experiment Result
### Training segmentation model with synthetic and real CT scan

![image](https://github.com/user-attachments/assets/460cc168-4c77-4644-9859-e47b901cbea7)

# Future/Current Plans 
Despite our initial experiments showing that synthetic data generated through Variational Autoencoders (VAEs) did not significantly enhance the performance of our segmentation model, we remain committed to improving our approach. Moving forward, we plan to explore a variety of alternative methods to improve the quality and effectiveness of synthetic data. 
1. Adjust the parameters
2. Change the input and architecture of our VAE
3. Explore other generative techniques, such as:
   * Generative Adversarial Networks (GANs)
   * Diffusion model
4. Enhancing data complexity and variability

# Question for you!
### Which one is real, and which one is fake? 
![image](https://github.com/user-attachments/assets/e31d4904-bf89-41eb-8c9d-51f2f7b895ba)
