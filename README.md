# Research Proposal
[Improving Early Diagnosis of Pancreatic Cancer with Synthetic Data](https://drive.google.com/file/d/1-e8EIY0hvOhud3bIM0zMkr6vkCG-FE8k/view?usp=sharing)

# Pancreatic Image Modality Database

1. [Kaggle Pancreas CT](https://www.kaggle.com/datasets/salihayesilyurt/pancreas-ct)
   
2. [Biobank Pancreatic MRI](https://biobank.ndph.ox.ac.uk/showcase/label.cgi?id=131)
   
3. [Cancer Imaging Archive Pancreatic CT](https://nbia.cancerimagingarchive.net/nbia-search/)
   
4. [Multiplex-Immunoflourescent Staining of Rapid Autopsy Samples from Human Pancreatic Cancer at the Primary and Metastatic Sites](https://edrn-labcas.jpl.nasa.gov/labcas-ui/c/index.html?collection_id=Multiplex_IF_Staining_Pancreatic_Cancer)
   
5. [A large annotated medical image dataset for the development and evaluation of segmentation algorithms](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) Label unbalanced with large background, medium pancreas, and small tumor structures, 420 3D CT Scans



# Pancreatic Cancer Relevant Reading

1. [AI innovation inspires hope in early detection of pancreatic cancer](https://newsnetwork.mayoclinic.org/discussion/mayo-clinics-ai-innovation-inspires-hope-in-early-detection-of-pancreatic-cancer/#:~:text=In%20a%20recent%20breakthrough%2C%20Mayo,intervention%20can%20still%20promise%20a)

2. [Automated Artificial Intelligence Model Trained on a Large Data Set Can Detect Pancreas Cancer on Diagnostic Computed Tomography Scans As Well As Visually Occult Preinvasive Cancer on Prediagnostic Computed Tomography Scans](https://www.gastrojournal.org/article/S0016-5085(23)04958-2/fulltext)

3. [Pancreatic Cancer Detection on CT Scans with Deep Learning: A Nationwide Population-based Study](https://pubs.rsna.org/doi/10.1148/radiol.220152)

4. [A large annotated medical image dataset for the development and evaluation of segmentation algorithms](https://arxiv.org/abs/1902.09063)

5. [The Medical Segmentation Decathlon](https://www.nature.com/articles/s41467-022-30695-9)

6. [Early Detection and Localization of Pancreatic Cancer by Label-Free Tumor Synthesis](https://arxiv.org/abs/2308.03008)

7. [Semantic segmentation of pancreatic medical images by using convolutional neural network](https://www.sciencedirect.com/science/article/pii/S1746809421010557)

8. [AX-Unet: A Deep Learning Framework for Image Segmentation to Assist Pancreatic Tumor Diagnosis](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.894970/full)


# Tutorial
1. [PyTorch and Monai for AI Healthcare Imaging - Python Machine Learning Course](https://youtu.be/M3ZWfamWrBM?si=Jb128JhHg0UcZ8HE)

2. [Create Infinite Medical Imaging Data with Generative AI](https://youtu.be/YHTSdd8-bnc?si=2s2ncxpQwdmnxG7F)

3. [AI Applications in Medical Imaging Segmentation](https://youtu.be/ryUCJHk2ckU?si=2LbeIxvmWohfSxnY)

# Resources
1. [MONAI](https://monai.io/started.html)

2. [CUDA](https://developer.nvidia.com/cuda-downloads)

3. [cuDNN](https://developer.nvidia.com/cudnn)

4. [State-of-Art Model for Medical Imaging](https://paperswithcode.com/sota/medical-image-segmentation-on-kvasir-seg)

Image classification vs. Object Detection vs. Image Segmentation 
<img width="1814" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b342247c-b048-4383-b88e-d149e7fd69ad">

# Relevent Conference
1. WACV
2. MICCAI
3. [CVPR](https://cvpr.thecvf.com/)
4. ICIP
5. [CCSC](http://ccscse.org/conference.php?year=38th)


# Blog
1. [OpenCV](OpenCV)
2. [CNN](CNN)
3. [Data Covert & Preprocess](DataCovert&Preprocess.md)
4. [Training](Experiments/training.md)
5. [Four-Fold](Experiments/FourFold64.md)

Variational AutoEncoder (VAE)

Variational AutoEncoder is an autoencoder and a generative model with other architecture

The encoder takes input and turns it into some vector, and the decoder takes a vector and expands it to be the input sample. In autoencoder, we dont really care about what the output is. But the latent vector

Sampling 
<img width="688" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/fd6dd3da-7cd5-457a-9429-9aefadffa474">

with autoencoder, we are not able to generate an image because we are just randomly picking a vector from the distribution pool of the input

However, the Variational encoder helps us determine where to pick the useful vectors from the input distribution. 

<img width="410" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/09c9ce16-3344-47ca-8767-4a0cf25717da">

<img width="784" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/854505c9-d6df-43d1-88de-045bc8d0d3b1">

<img width="868" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/15a051f7-28e8-4605-bf37-5796b7613974">

<img width="844" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1a7fd5bf-0d7c-47cf-a299-80e066a7a56d">

<img width="859" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/ea5b0e9b-68e8-4c98-a31d-61a019e2b081">



