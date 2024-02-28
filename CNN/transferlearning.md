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
 
# Code - Feature Extraction

    class ImagenetTransferLearning(pl.LightningModule):
        def __init__(self):
            super().__init__()
    
            self.accuracy = torchmetrics.Accuracy()
    
            # init a pretrained resnet
            # load the ResNet model as backbone
            backbone = models.resnet50(pretrained=True)

            # Get the all the layers except the last one, but the last one has 1000 outputs
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
            
            # use the pretrained model
            # Now we just add the last layer with 2 output to the network 
            num_target_classes = 2
            self.classifier = nn.Linear(num_filters, num_target_classes)
    
        def forward(self, x):
            # Change our forward function to include the 4 lines below
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
            x = self.classifier(representations)
            return F.softmax(x,dim = 1) 

# Code - Fine Tune

    # We load a pretrained resnet18 model and change the final fully connected layer to output to our class size (2).
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

Here, we need to freeze all the network layers except the final layer.
We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().

    model_conv = torchvision.models.resnet18(pretrained=True)
    
    # We freeze layers here
    for param in model_conv.parameters():
        param.requires_grad = False
    
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    
    model_conv = model_conv.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

