# implementation of Four-Fold

  1. I first create four folds in my computer and separate the data into the 4 folders. However, the number of data in each folder is not exactly the same. but they are around 64 nifti.gz file in each folder.
  2. Then I changed the path for the data as the following

    train_data1 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\train\Fold1\*nii.gz"))
    label_data1 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\label\Fold1\*nii.gz"))

    train_data2 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\train\Fold2\*nii.gz"))
    label_data2 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\label\Fold2\*nii.gz"))

    train_data3 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\train\Fold3\*nii.gz"))
    label_data3 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\label\Fold3\*nii.gz"))

    train_data4 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\train\Fold4\*nii.gz"))
    label_data4 = sorted(glob(r"D:\LumenResearchDataBase\DataBase\group64\nifti_files\label\Fold4\*nii.gz"))
    
    train_file_part1 = [{"img": image, "seg": label} for image, label in zip(train_data1, label_data1)]
    train_file_part2 = [{"img": image, "seg": label} for image, label in zip(train_data2, label_data2)]
    train_file_part3 = [{"img": image, "seg": label} for image, label in zip(train_data3, label_data3)]

    # Then, concatenate these lists to form a single training file
    train_file = train_file_part1 + train_file_part2 + train_file_part3

    test_file = [{'img':image, 'seg':label} for image, label in zip(train_data4, label_data4)]

  3. Then, I change the path for test data and training data to different folders. 

# Experiement with the learning rate and weight decay

Epoch_loss: 0.0056

Epoch_metric:  0.9944

test_loss_epoch: 0.2967

test_dice_epoch: 0.7033

current epoch: 600 current mean dice: 0.6396

best mean dice: 0.7424 at epoch: 57

train completed, best metric: 0.7424 at epoch: 57

<img width="770" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/70e68652-c4ce-46c1-bad7-fa8292ccbb91">

<img width="372" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/caf29e4a-ae42-49c5-82af-15f254513c5a">

epoch 600/600
____________________
Epoch_loss: 0.0056

Epoch_metric:  0.9944

test_loss_epoch: 0.2967

test_dice_epoch: 0.7033

current epoch: 600 current mean dice: 0.6396

best mean dice: 0.7424 at epoch: 57

train completed, best metric: 0.7424 at epoch: 57

<img width="927" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/1cca5342-1dfb-4a51-9565-17920d43cfc0">

<img width="371" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/38fb8981-ceaa-44ee-bcf5-3b54ffeefd43">


# Four Fold with groups of 64 slices

Epoch_loss: 0.0022

Epoch_metric:  0.9978

test_loss_epoch: 0.3125

test_dice_epoch: 0.6875

current epoch: 600 current mean dice: 0.7625

best mean dice: 0.7421 at epoch: 25

train completed, best metric: 0.7421 at epoch: 25

<img width="652" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/0cee0d66-fd12-4dc7-9d20-cf0dfd1cf12e">


<img width="560" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/98889315-eba2-4117-b0b9-60efcc5ec315">

Epoch_loss: 0.0017

Epoch_metric:  0.9983

test_loss_epoch: 0.3118

test_dice_epoch: 0.6882

current epoch: 600 current mean dice: 0.7259

best mean dice: 0.7256 at epoch: 24

train completed, best metric: 0.7256 at epoch: 24

<img width="646" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/c9bd2f6d-20e2-4cf5-8b52-82ed10f50f0c">


Epoch_loss: 0.0036

Epoch_metric:  0.9964

test_loss_epoch: 0.3139

test_dice_epoch: 0.6861

current epoch: 600 current mean dice: 0.7329

best mean dice: 0.7362 at epoch: 31

train completed, best metric: 0.7362 at epoch: 31

<img width="652" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/76dca5f2-f2c2-4e8e-bd55-680dbddb610e">

Epoch_loss: 0.0025

Epoch_metric:  0.9975

test_loss_epoch: 0.3064

test_dice_epoch: 0.6936

current epoch: 600 current mean dice: 0.7656

best mean dice: 0.7295 at epoch: 29

train completed, best metric: 0.7295 at epoch: 29

<img width="825" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/6e080c30-4160-4e02-858c-7f058f1f08d2">


# Result of Four-Fold
Experiment 1: 0.7421 accuracy

Experiment 2: 0.7256 accuracy

Experiment 3: 0.7362 accuracy

Experiment 4: 0.7295 accuracy

Average accuracy: (0.7421 + 0.7256 + 0.7362 + 0.7295) / 4 = 0.73335


# Reflection
After I experienced the four-fold method, I got an average accuracy of around 0.73%, which is not great. Therefore, I think it can be improved in different ways
1. Get more data
2. Try with a different model
3. Try different transform methods (How to change the intensity so the pancreas shows up more)

Those are the ways that I can think of so far to improve the performance. I think increasing the number of epochs won't help much. The model might tend to be overfitted with 600 epochs. since the accuracy is already around 99%. 

Also, something that I noticed is the best metric is always found when the epochs are less than 100. Does that mean the model also learned well enough with 100 epochs? 

# Question
Issues with the testing loss

Should We register for CVPR?

What is the difference between a summit and a conference?
https://genaisummit.ai/#/
  - Summit is more a business side of things; people can look for venture capital. People will present more innovative business ideas. On the other hand conference is more academic. 

Why is a computer vision based on RBG? There are more colors in the world than even humans can't see. Is there are people working on seeing more color to get more information? 
  - Computer vision can work more than just RBG. In the camera, when we capture things. There will be sensors that can capture how much light reflects on the red sensor, etc. We can also add more sensors to detect different things that might reflect from the light. Even in medical images, what we are working on is not RBG. 






