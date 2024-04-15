# Combined Dataset

### Create a dictionary that store both datasets, and shuffle them

    files_dataset1 = [{'train': train, 'label': label} 
                      for train, label in zip(sorted(glob(os.path.join(train_dataset1_path, "*"))), 
                                              sorted(glob(os.path.join(label_dataset1_path, "*"))))]
    
    files_dataset2 =[{'train': train, 'label': label} 
                      for train, label in zip(sorted(glob(os.path.join(train_dataset2_path, "*"))), 
                                              sorted(glob(os.path.join(label_dataset2_path, "*"))))]

    files = files_dataset1 + files_dataset2
    random.shuffle(files)

### Create new dataset from files, and them those combined and shuffled dataset into a new directory 

    train_path = r'D:\LumenResearchDataBase\DataBase\combine\CombinedOriginal\Combined\train'
    label_path = r'D:\LumenResearchDataBase\DataBase\combine\CombinedOriginal\Combined\label'
    
    for idx, file in enumerate(files, start=0):
        
        # New file name with "Pancreas_" prefix and an index
        file_name = f"Pancreas_{idx}"
        train_new_path = os.path.join(train_path, file_name)
        label_new_path = os.path.join(label_path, file_name)
        
        os.mkdir(train_new_path)
        os.mkdir(label_new_path)
    
        for train in glob(os.path.join(file['train'], '*')):
            shutil.move(train, train_new_path)
            
        for label in glob(os.path.join(file['label'], '*')):
            shutil.move(label, label_new_path)

# Train the model with combined dataset

      train_data, test_data = perpare(data_dir, (1.5, 1.5, 1.0), -50, 200, [128, 128, 64], True)
      optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-3, amsgrad=True)

Epoch_loss: 0.0178
Epoch_metric:  0.9822
test_loss_epoch: 0.1635
test_dice_epoch: 0.8365
current epoch: 250 current mean dice: 0.9166
best mean dice: 0.8716 at epoch: 24
train completed, best metric: 0.8716 at epoch: 24

<img width="758" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/8ce103f8-3e32-4796-8925-49fbfb46876c">


<img width="374" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/0e5bb6af-bce1-4cb6-ab2c-6463ac7b81ae">

<img width="378" alt="image" src="https://github.com/tan200224/Research_Blog/assets/68765056/e1ce0bc5-fb22-420c-8c54-f23d2068fb35">

# Result 
Even though there are some differences in the dataset, the accuracy of the model is still improved. The model is a little bit overfit. 

