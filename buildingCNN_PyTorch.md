# The Frame work of buliding a CNN
1. Import Pytorch Library and functions
2. Define Transformer
3. Load dataset
4. Inspect and Visualize image dataset
5. Create data loader for load batches of images
6. Builiding Model
7. Training Model
8. Analyizing it's Accuracy
9. Saving Model
10. Plotting trainging logs


## Import libaries and modules

    # Import PyTorch
    import torch
    
    # Use torchvivision to get datset and useful image transformations
    import torchvision
    import torchvision.transforms as transform

    # Import PyTorch's optimization library and nn
    # nn is used as the basic building block for network graphs
    import torch.optim as optim
    import torch.nn as nn

    # Check if we are using the GPU
    print("GPU available: {}".format(torch.cuda.is_available()))



## Define transformer
Transformers are needed to cast the image data into the required format for input into the model.

We use transform.ToTensor() to convert the image dat into a PyTorch Tensor
We use transform.Normalize() to normalize  the pixel values

TO Normalize the image data between -1 and +1, pass the input as (0.5, ) (0.5, ). For RGB images the input will be transformed.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

### Why Normalize?
1. To ensure all feature and pixel intensities are weighted equally
2. Makes training faster as it avoids oscillations during training
3. Removes bias or skewness in the image data

### Why 0.5?
The normalization is done like this:
image = (image - mean) / std
**Using the parameters 0.5 and 0.5 sets the Mean and STD to 0.5. Using the formula above gives us:
* Min value = (0-0.5)/0.5 = 1
* Max value = (1-0.5)/0.5 = -1

      # Transform to a PyTorch tensor and the normalize our value between -1 and +1
      transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, ), (0.5, ))])


## Fetch MNIST Dataset using torchvision
* Transforms will not applied upon loading, It is only applied when loaded by our data
* The dataset is left unchanged, only batches of images loaded by the data loader are copied and transformed every iteration

      # Load training data and specify what transform to use when loading
      trainset = torchvision.dataset.MNIST('mnist',
                                            train = True,
                                            dowload = Ture,
                                            transform = transform

      # Load our test data
      testset = torchvision.dataset.MNIST('mnist',
                                          train = False,
                                          download = True
                                          transform = transform
  
### About the training and Test data
There are two subsets of the data being used here:
* Training data data that is used to optimize model parameters (used during training)
* Test/Validation data Data that is used to evaluate the model performance


## Inspect a sample training data

      print(trainset.data.shape)
      print(testset.data.shape)

      print(train.data[0].shape)
      print(train.data[0])


### we can also plot those data into OpenCV
But we need to convert those tensor to a NumPy array

    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    # Define our imshow function
    def imgshow(title="", image = None, size = 6):
        w, h = image.shape[0], image.shape[1]
        aspect_ratio = w/h
        plt.figure(figsize=(size * aspect_ratio,size))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()
    
    # Convert image to a numpy array
    image = trainset.data[0].numpy()
    imgshow("MNIST Sample", image)

<img width="270" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/205de179-d215-42a5-a51f-347cb973d0f4">

### We can also use matplotlib to show many examples from the dataset
    # Let's view the 50 first images of the MNIST training dataset
    import matplotlib.pyplot as plt
    
    figure = plt.figure()
    num_of_images = 50
    
    for index in range(1, num_of_images + 1):
        plt.subplot(5, 10, index)
        plt.axis('off')
        plt.imshow(trainset.data[index], cmap='gray_r')


## Create data loader
A Data loader is a function that we can use to grab a specified batch size of data during training.
** We can't feed all the data into the network at once, therefore that is why we need to split data into batches.**
We set **shuffle** to True to prevent data sequence bias
**num_worker** specifies how many CPU cores we want to utilize. Usually set it to 0.

      # Perpare train and test loader
      trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size = 128,
                                                shuffle = True,
                                                num_workers= 0)

      testloader = torch.utils.data.DataLoader(testset,
                                                batch_size = 128,
                                                shuffle = False,
                                                num_workers= 0)

                                            

### Using Iter and Next() for load batches

    # we use Python function iter to return an iterator for our train_loader object
    dataiter = iter(trainloader)

    # We use next to get the first batch of data from our iterator
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)

torch.Size([128, 1, 28, 28])
torch.Size([128])


### PyTorch Image Plotting Tool


    import matplotlib.pyplot as plt
    import numpy as np
    
    # function to show an image
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    
    # show images
    imshow(torchvision.utils.make_grid(images))
    
    # print labels
    print(''.join('%1s' % labels[j].numpy() for j in range(128)))


## Build the model

### Building a Convolution Filter Layer

nn.Conv2d(in_channels=1,
          out_channels=32,
          kernel_size=3,
          stride=1,
          padding=1)

* in_channels (int) — This is the number of channels in the input image (for grayscale images use 1 and for RGB color images use 3)
* out_channels (int) — This is the number of channels produced by the convolution. We use 32 channels or 32 filters. NOTE 32 will be the number of in_channels in the next network layer.
* kernel_size (int or tuple) — This is the size of the convolving kernel. We use 3 here, which gives a kernel size of 3 x 3.
* stride (int or tuple, optional) — Stride of the convolution. (Default: 1)
* padding (int or tuple, optional) — Zero-padding added to both sides of the input (Default: 0). We use a padding = 1.


![image](https://github.com/tan200224/Blog/assets/68765056/fb653c0e-1a22-4478-8a26-aec6487e248f)

### What is torch.nn functional?
It is usually imported into the namespace F by convention. It contains all the functions in torch.nn library, as well as a wide range of loss and activation functions.

      import torch.nn as nn
      import torch.nn.functional as F

      # Create our model using python class
      class Net(nn.Module)
        def __init__(self):
          # super is a subclass of the nn.Module and inherits all its methods
          super(Net, self).__init__()

          # We define our layer objects here
          First CNN Layer using 32 Filters of 3x3 size, with stride of 1 and padding of 0
          self.conv1 = nn.Conv2d(1,32,3)

          #Second CNN Layer using 64 Filters of 3x3 size, with stride of 1 and padding of 0
          self.conv2 = nn.Conv2d(32,64,3)

          # Our Max Pool Layer 2x2 kernel of stride 2
          self.pool = nn.MaxPool2d(2,2)

          # The first fully connected layer, connects output of Max Pool
          # which is 12 x 12 x 64 and connects the 128 nodes to 10 output nodes
          self.fc1 = nn.Linear(64 * 12 * 12, 128)

          # Second Fully Connected Layer, connects the 128 nodes to 10 output nodes
          self.fc2 = nn.Linear(128, 10)


        def forward(self, x):
          # Define our forward Propagation sequence
          # with the order of Conv1 - ReLu - Conv2 - ReLU - Max Pool - Flatten - FC1 - FC2
          x = F.relu(self.conv1(x))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.view(-1, 64*12*12) # Flatten
          x = F.relu(self.fc1(x))
          x = self.fc2(x)

    net = Net()
    net.to(device)
    print(net)
    
Net(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)



## Defining a Loss Function and Optimizer

    # We import our optimizer function
    import torch.optim as optim
    
    # We use Cross Entropy Loss as our loss function
    criterion = nn.CrossEntropyLoss()
    
    # For our gradient descent algorthim or Optimizer
    # We use Stochastic Gradient Descent (SGD) with a learning rate of 0.001
    # We set the momentum to be 0.9
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



## Training our Model

![image](https://github.com/tan200224/Blog/assets/68765056/a0d81528-86c2-4957-954f-92efd0e8c033)

    # We loop over the traing dataset multiple times (each time is called an epoch)
    epochs = 10
    
    # Create some empty arrays to store logs
    epoch_log = []
    loss_log = []
    accuracy_log = []
    
    # Iterate for a specified number of epochs
    for epoch in range(epochs):
        print(f'Starting Epoch: {epoch+1}...')

         # We keep adding or accumulating our loss after each mini-batch in running_loss
      running_loss = 0.0
  
      # We iterate through trainloader iterator
      # Each cycle is a minibatch
      for i, data in enumerate(trainloader, 0): # Why is there a 0?
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
  
          # Move our data to GPU
          inputs = inputs.to(device)
          labels = labels.to(device)
  
          # Clear the gradients before training by setting to zero
          # Required for a fresh start
          optimizer.zero_grad()
  
          # Forward -> backprop + optimize
          outputs = net(inputs) # Forward Propagation
          loss = criterion(outputs, labels) # Get Loss (quantify the difference between the results and predictions)
          loss.backward() # Back propagate to obtain the new gradients for all nodes
          optimizer.step() # Update the gradients/weights
  
          # Print Training statistics - Epoch/Iterations/Loss/Accuracy
          running_loss += loss.item()
          if i % 50 == 49:    # show our loss every 50 mini-batches
              correct = 0 # Initialize our variable to hold the count for the correct predictions
              total = 0 # Initialize our variable to hold the count of the number of labels iterated
  
              # We don't need gradients for validation, so wrap in
              # no_grad to save memory
              with torch.no_grad():
                  # Iterate through the testloader iterator
                  for data in testloader:
                      images, labels = data
                      # Move our data to GPU
                      images = images.to(device)
                      labels = labels.to(device)
  
                      # Foward propagate our test data batch through our model
                      outputs = net(images)
  
                       # Get predictions from the maximum value of the predicted output tensor
                       # we set dim = 1 as it specifies the number of dimensions to reduce
                      _, predicted = torch.max(outputs.data, dim = 1)
                      # Keep adding the label size or length to the total variable
                      total += labels.size(0)
                      # Keep a running total of the number of predictions predicted correctly
                      correct += (predicted == labels).sum().item()
  
                  accuracy = 100 * correct / total
                  epoch_num = epoch + 1
                  actual_loss = running_loss / 50
                  print(f'Epoch: {epoch_num}, Mini-Batches Completed: {(i+1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                  running_loss = 0.0
  
      # Store training stats after each epoch
      epoch_log.append(epoch_num)
      loss_log.append(actual_loss)
      accuracy_log.append(accuracy)

print('Finished Training')



