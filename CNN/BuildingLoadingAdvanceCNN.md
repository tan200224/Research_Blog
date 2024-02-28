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


### Process the data and put them into the model
    import torch
    import json
    from PIL import Image
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import numpy as np
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('imageNetclasses.json') as f:
      class_name = json.load(f)
    
    def predict_image(images, class_names):
      to_pil = transforms.ToPILImage()
      fig = plt.figure(figsize=(16,16))
    
      for (i, image) in enumerate(images):
        # Convert to iamge and tensor

        # convert the images to a PIL image format firstly
        image = to_pil(image) 

        # We convert those images to float
        image_tensor = test_transforms(image).float() 

        # We make this to a one dimensional tensor
        image_tensor = image_tensor.unsqueeze_(0) 
        
        # We convert the tensor to a pytorch variable, this will allow us to forward propgate the network
        input = Variable(image_tensor) 
        output = input.to(device)
        output = model(input)

        # We convert the data into numpy array using output.cpu().numpy(), then we get the index with the highest probability class by using .argmax()
        index = output.data.cpu().numpy().argmax() 
        name = class_name[str(index)]
    
        # plot image
        sub = fig.add_subplot(len(images), 1, i+1)
        sub.set_title(f"Predicted {str(name)}")
        plt.axis("off")
        plt.imshow(image)
      plt.show()
    
    
    def get_images(directory = './images'):
      data = datasets.ImageFolder(directory, transform=test_transforms)
      num_images = len(data)
      loader = torch.utils.data.DataLoader(data, batch_size=num_images)
      dataiter = iter(loader)
      images, labels = next(dataiter)
      return images


## Rank-N
* Rank-N Accuracy is a way to measure a classifier's accuracy with a bit more leeway.

![image](https://github.com/tan200224/Blog/assets/68765056/43c55a03-7171-4053-8d8c-343aeabd2168)

* it will return "Shetland sheepdog" as the predicted class (highest probability). However, that would be incorrect
* The correct class, "Collie" is actually the second most probable, which means the classifier is still doing quite well, but it is not reflected if we only look at the top predicted class.
* For example, Rank-5, will consider any of the top 5 most likely classes for the predicted label
  
To get the Rank-N accuracy, instead of using argmax(), which give us the highest probability, we will use torch.nn.functional.toppk(num of rank, dimension) 

    import torch.nn.functional as nnf

    # we the the softmax probablitiy from the output, perviously we only used argmax to get the top probability
    prob = nnf.softmax(output, dim=1) 

    # we the the number of rank we want, and the dimension
    top_p, top_class = prob.topk(5, dim=1)
    print(top_p, top_class)

### create a function that get us the class names
    def getClassNames(top_classes):
      top_classes = top_classes.cpu().data.numpy()[0]
      all_classes = []
      for top_class in top_classes:
        all_classes.append(class_names[str(top_class)])
      return all_classes
      
    getClassNames(top_class)

### Construct our function to give us our Rank-N Accuracy
    from os import listdir
    from os.path import isfile, join
    import matplotlib.pyplot as plt
    
    fig=plt.figure(figsize=(16,16))
    
    def getRankN(model, directory, ground_truth, N, show_images = True):
      # Get image names in directory
      onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    
      # We'll store the top-N class names here
      all_top_classes = []
    
      # Iterate through our test images
      for (i,image_filename) in enumerate(onlyfiles):
        image = Image.open(directory+image_filename)
    
        # Convert to Tensor
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model(input)
        # Get our probabilties and top-N class names
        prob = nnf.softmax(output, dim=1)
        top_p, top_class = prob.topk(N, dim = 1)
        top_class_names = getClassNames(top_class)
        all_top_classes.append(top_class_names)
    
        if show_images:
          # Plot image
          sub = fig.add_subplot(len(onlyfiles),1, i+1)
          x = " ,".join(top_class_names)
          print(f'Top {N} Predicted Classes {x}')
          plt.axis('off')
          plt.imshow(image)
          plt.show()
    
      return getScore(all_top_classes, ground_truth, N)
    
    def getScore(all_top_classes, ground_truth, N):
      # Calcuate rank-N score
      in_labels = 0
      for (i,labels) in enumerate(all_top_classes):
        if ground_truth[i] in labels:
          in_labels += 1
      return f'Rank-{N} Accuracy = {in_labels/len(all_top_classes)*100:.2f}%'














