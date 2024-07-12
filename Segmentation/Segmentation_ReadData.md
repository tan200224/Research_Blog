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
![Screenshot 2024-07-12 181117](https://github.com/user-attachments/assets/6fa336b5-e7d5-468c-87fa-a2dd02238590)
![Screenshot 2024-07-12 181348](https://github.com/user-attachments/assets/070c6b2e-a31b-496e-8322-325728eeb190)

![Screenshot 2024-07-12 181420](https://github.com/user-attachments/assets/a709f4e8-cb79-481d-bb24-6180984667aa)











