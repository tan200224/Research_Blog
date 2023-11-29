# What Are GANs?
First introduced in 2014 by lan Goodfellow et al. GANs are a type of Neural Network that generates data that plausible comes fomr an existing dirbution of samples.

### Examples of GANs

### Realistic Photo Gerneration

<img width="565" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/c240d6d5-306b-4744-8b75-397b977f5a70">

### Image-to-Image Translation

<img width="556" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/7a935e1d-a4e0-48a0-828a-5774826065d7">

### Text-to-Image Translations

<img width="497" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/71fcb27f-b934-4fd5-a9a6-3448fe28d7eb">

### Semantic-Image-to-Photo Translation

<img width="728" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/0c27a28c-2d52-4f59-8f08-52bff752a186">

### Super Resolution (SRGAN)

<img width="678" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/8c370a87-1430-4d26-9ad2-2b4dc5268083">


# Training GANs
* Training GANs is notoriously difficult compared to Neural Networks we use gradient descent to change our weights to reduce our loss.
* In a GANs, every weight change can change the entire balance of our dynamic system.
* We are not seeking to minimize loss, but finding an equilibrium between our two opposing Networks
* Training stops when the Discriminator can not tell apart Real vs Fake Data

## The Training Process
1. We randomly generate a noisy vector
2. input this into our Generator Network to generate sample data
3. We then take some sample data from our real data and mix it with some of our generated data
4. Train our discriminator to classify this mixed dataset and thus update its weight accordingly
5. Now we train the generator, make more random noisy vectors, and create synthetic data. With the weights of the discriminator frozen, we use the feedback from the discriminator to update the weights of the generator. 

## Are There Any Issues With GANs?

<img width="532" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/b304645a-8de6-48c2-aaeb-e6d12075e951">

## Challanges in training 
* Achieving Equilibrium
* Time
* Bad Initializations - Caseing the discriminator loss to go close to zero
* Mode Collapse - This happens when, regardless of the noise input fed into your generator, the generated output varies very little. It occurs when a small set of images looks good to the discriminator and get scored better than other images. The GAN simply learns to reproduce those images over and over.


# Use Case of GANs

<img width="569" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/a06c3fb9-50ae-431a-ae94-efb2f4fff3fc">


## GAN on MNIST Dataset

### 1. Setting up our data and modules
    import torch
    from torch import nn
    
    import math
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms as transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(42)


### 2. Fetch our MNIST Dataset using torchvision and Create our transforms and Data Loader

    batch_size = 32
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    
    train_set = torchvision.datasets.MNIST(root="", train=True, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)


### Plot and take a look our the original data

    samples, data = next(iter(train_loader))
    
    for i in range(32):
        ax = plt.subplot(4, 8, i+1)
        plt.imshow(samples[i].reshape(28,28), cmap='gray')
        plt.xticks([])
        plt.yticks([])

<img width="386" alt="image" src="https://github.com/tan200224/Blog/assets/68765056/754c6f2f-4d6e-44c2-bd98-52b27ea3a656">


### 3. Define our Discriminator Model
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(784, 2048),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            x = x.view(x.size(0), 784)
            output = self.model(x)
            return output
    
    discriminator = Discriminator().to(device=device)

### 4. Define our Generator Model
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(100, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 784), # The final image has 784 pixel
                nn.Tanh(), # We use the Tanh() activation function so that our put lie between 1 and -1
            )
            
        def forward(self, x):
            output = self.model(x)
            output = output.view(x.size(0), 1, 28, 28) # we are reshaping the output from 784 to 28*28
            return output
    
    generator = Generator().to(device=device)


### Set training parameter
    lr = 0.0001
    epoches = 500
    loss_function = nn.BCELoss()
    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


### Training 
    for epoch in range(epoches):
        
        # Get data for trainign the discriminator
        for n, (samples, labels) in enumerate(train_loader):
            
            real_samples = samples.to(device=device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
            
            # Generate a random vector
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
            
            generated_samples = generator(latent_space_samples)
            
            # Make the fake image label to 0
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            
            
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            
            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
            
            # Training the generator 
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
            
            if n == batch_size - 1:
                print(f'Epoch: {epoch} Descriminated Loss: {loss_discriminator}')
                print(f'Epoch: {epoch} Generator Loss: {loss_generator}')


### 5. Inspect our generated Samples

    # Once our generator is trained, we want it to produce some image, so we can take a look at it
    
    latent_space_samples = torch.randn(batch_size, 100).to(device=device)
    generated_sample = generator(latent_space_samples)

    generated_samples = generated_sample.cpu().detach()
    
    for i in range(16):
        ax = plt.subplot(4, 4, i+1)
        plt.imshow(generated_samples[i].reshape(28,28), cmap='gray')
        plt.xticks([])
        plt.yticks([])
    

    
