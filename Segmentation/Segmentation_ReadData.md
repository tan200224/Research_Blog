## This is the experinment run with slices of 4 groups. It contains the following dataset: 
1. Kaggle (Cancer Imaging Archive Pancreatic CT)
2. A large annotated medical image dataset for the development and evaluation of segmentation algorithms Label
3. Combined Dataset


# Data process

    def example_transform(data, label):
        # Apply any preprocessing here
        data = data[30:-30, 100:-100, :]
        label = label[30:-30, 100:-100, :]
        data = np.clip(data, -50, 200)
        min_val = data.min()
        max_val = data.max()
        data = (data - min_val) / (max_val - min_val)
        data = torch.tensor(data).permute(2, 0, 1).unsqueeze(0)
        label = torch.tensor(label).permute(2, 0, 1).unsqueeze(0)
        
        data = TF.resize(data, size=(256, 256), interpolation=TF.InterpolationMode.BILINEAR)  # For training data
        label = TF.resize(label, size=(256, 256), interpolation=TF.InterpolationMode.NEAREST)  # For label data
    
        # Permute dimensions for PyTorch compatibility (1, Channel, Height, Width)
        data = data.permute(0, 2, 3, 1)
        label = label.permute(0, 2, 3, 1)
        
        
        return data, label

# Model Setup

    def dice_metric(predicted, target):
        dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
        value = 1-dice_value(predicted, target).item()
        return value

        
    model = UNet(
        spatial_dims = 3,
        in_channels = 1,
        out_channels = 2,
        channels = (16, 32, 64, 128, 256),
        strides = ((2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
        num_res_units = 2,
        norm = Norm.BATCH
    ).to(device)

    max_epochs = 200
    loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    optim = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-4, amsgrad=True)
    model_dir = r"D:\LumenResearchDataBase\DataBase\task07\group4\model"
    test_interval= 1
    device=torch.device('cuda:0')
    


# Result:
## A large annotated medical image dataset
![Screenshot 2024-07-12 215216](https://github.com/user-attachments/assets/45f02539-fb70-4719-be00-ccacee3c1d0c)
![Screenshot 2024-07-12 215348](https://github.com/user-attachments/assets/13b012ea-9c5d-4b56-85c7-f84145fdd8a5)

![Screenshot 2024-07-12 215453](https://github.com/user-attachments/assets/e569312d-6123-49d1-ad5f-78258046a73b)


## Kaggle
![image](https://github.com/user-attachments/assets/bb03301e-caf6-4385-8b0c-d49733f48b6b)
![Screenshot 2024-07-12 201759](https://github.com/user-attachments/assets/0546b682-ec13-49fe-b896-5733c0ecd261)

![Screenshot 2024-07-12 201826](https://github.com/user-attachments/assets/81853e73-51ee-45d2-927d-e4332cfd6993)

## Combined
![Screenshot 2024-07-13 011523](https://github.com/user-attachments/assets/fc15bc34-16c3-4033-b440-3a220cac1b49)
![Screenshot 2024-07-13 011820](https://github.com/user-attachments/assets/b992a30d-8c13-401c-846c-aa22554b3848)

![Screenshot 2024-07-13 011923](https://github.com/user-attachments/assets/a8ef959b-575c-4934-a6af-9ea23a3214dd)






