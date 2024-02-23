# Coverting Data between DICOM and NIFTI, and Create Groups of Slices

## The Library we use

    import pandas as pd
    import numpy as np
    import cv2 
    from matplotlib import pyplot as plt
    from glob import glob
    import shutil
    import os
    import dicom2nifti
    from tqdm import tqdm

## Initial the input path and output path

    in_path = r"D:\LumenResearchDataBase\Task07_Pancreas\Task07_Pancreas\dicom_ungroup\labels"
    out_path = r"D:\LumenResearchDataBase\Task07_Pancreas\Task07_Pancreas\dicom_files\labels"

## Moving the Slices in to groups

    for patient in glob(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))
        
        # Here we need to calculate the number of folders which mean into how many groups we will divide the number of slices
        Number_slices = 90
        number_folders = int(len(glob(patient + '/*')) / Number_slices)
    
        for i in range(number_folders):
            output_path = os.path.join(out_path, patient_name + '_' + str(i))
            os.mkdir(output_path)
    
            # Move the slices into a specific folder so that you will save memory in your desk
            for i, file in enumerate(glob(patient + '/*')):
                if i == Number_slices + 1:
                    break
                    
                shutil.move(file, output_path)
                print("moved")

## Convert the dicom files into nifties

    in_path_training = r'D:\LumenResearchDataBase\Task07_Pancreas\Task07_Pancreas\dicom_files\training\*'
    in_path_labels = r'D:\LumenResearchDataBase\Task07_Pancreas\Task07_Pancreas\dicom_files\labels\*'
    
    out_path_training = r'D:/LumenResearchDataBase/Task07_Pancreas/Task07_Pancreas/nifti_files/training'
    out_path_labels = r'D:/LumenResearchDataBase/Task07_Pancreas/Task07_Pancreas/nifti_files/labels'
    
    list_training = glob(in_path_training)
    list_labels = glob(in_path_labels)
    
    
    for patient in list_training:
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(list_training[0], out_path_training, patient_name + '.nii.gz')
        
    for patient in list_labels:   
        patient_name = os.path.basename(os.path.normpath(patient))
        dicom2nifti.dicom_series_to_nifti(patient, os.path.join(out_path_labels, patient_name + ".nii.gz"))


## Bug

'<' not supported between instances of 'NoneType' and 'NoneType'



## PreProocessing(Transform and Load the data)


## The library used

    import os
    from glob import glob
    
    import torch
    from monai.transforms import(
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Spacingd,
        ScaleIntensityRanged,
        CropForegroundd,
        Resized,
        ToTensord
    )
    
    from monai.data import Dataset, DataLoader
    from monai.utils import first
    
    from matplotlib import pyplot as plt



## Initial the path and zip the image and label in to a dictionary

    train_images = sorted(glob(os.path.join(data_dir, "imagesTr", "*nii.gz")))
    labeled_images = sorted(glob(os.path.join(data_dir, "labelsTr", "*nii.gz")))
    
    # Valiation data
    
    train_files = [{"image":image_name, "label":label_name} for image_name, label_name in zip(train_images, labeled_images)]
    
    train_files

## Transform

    # Load the image
    # Add channl: batch
    # zoom in the image, and make the pixel weight more, [col, row, depth]
    # Change the intensity range of the image (change the contrast , normalize the value between 0.0 to 1.0 (not o and 1))
    # Crop only the forground (you want to  crop both image and label, source_key: which image it should crop based on)
    # Size the image because patients might have different size of image. Note: do it after the crop. Spatial_size[W, H, Slice]
    # Convert to tensor
    
    original_transform = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(['image', 'label']),
            ToTensord(keys=['image', 'label'])
        ]
    )
    
    
    train_transform = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            EnsureChannelFirstd(['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
            ScaleIntensityRanged(keys='image', a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key = 'image'),
            
            # Poteintial Error: The number of slice must be consistent
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 128]),
            ToTensord(keys=['image', 'label'])
        ]
    )


## Load the data

train_ds = Dataset(data=train_files, transform=train_transform)
train_loader = DataLoader(train_ds, batch_size=1)

original_ds = Dataset(data=train_files, transform=original_transform)
oringal_loader = DataLoader(original_ds, batch_size=1)

### Get the first batch

test_patient = first(train_loader)
oringal_patient = first(oringal_loader)


## Visualize the data

<img width="737" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b7f54b99-4bc3-4ae3-8e45-ae3cc8c228ac">


## Question

What is the Zip do? 
