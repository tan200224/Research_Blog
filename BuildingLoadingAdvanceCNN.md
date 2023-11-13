# Loading Pre-trained Network in PyTorch (ResNets, DenseNets, MobileNET, VGG19)

View all the Models Available in PyTorch here - https://pytorch.org/vision/main/models.html

    import torchvision.models as models
    model = models.vgg16(pretrained=True)
    print(model)
    
<img width="621" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/e276aa30-eb43-43a3-a4ed-697ba06f1d1f">


### Check the number of parameters 

    from torchsummary import summary 
    summary(model, input_size = (3,224,224))

<img width="476" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/2a63a5e4-3385-4f94-b6c7-a0d4e3ca51c2">


All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. You can use the following transform to normalize:
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    from torchvision import datasets, transforms, models

    data_dir = '/images'

    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      #transforms.Normalize([0.485, 0.456, 0.406],
                                      #                     [0.229, 0.224, 0.225])
                                     ])

### Net.eval()
net.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:

    model.eval()



