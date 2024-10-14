# Generating Synthetic Medical CT Scans using GANs

Compared to VAE, using GANs is harder in terms of training. 
GAN took longer time to finished the training, and it is hard to determine when would be a great stopping point for the training 

### Generator Model:
      self.main = nn.Sequential(
        # Input layer: Upsample from Z_DIM to 4x4 spatial resolution
        nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 16, 4, 1, 0, bias=False),
        nn.BatchNorm2d(G_HIDDEN * 16),
        nn.ReLU(True),
        # 1st hidden layer: Upsample to 8x8
        nn.ConvTranspose2d(G_HIDDEN * 16, G_HIDDEN * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(G_HIDDEN * 8),
        nn.ReLU(True),
        # 2nd hidden layer: Upsample to 16x16
        nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(G_HIDDEN * 4),
        nn.ReLU(True),
        # 3rd hidden layer: Upsample to 32x32
        nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(G_HIDDEN * 2),
        nn.ReLU(True),
        # 4th hidden layer: Upsample to 64x64
        nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
        nn.BatchNorm2d(G_HIDDEN),
        nn.ReLU(True),
        # 5th hidden layer: Upsample to 128x128
        nn.ConvTranspose2d(G_HIDDEN, G_HIDDEN // 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(G_HIDDEN // 2),
        nn.ReLU(True),
        # Output layer: Upsample to 256x256
        nn.ConvTranspose2d(G_HIDDEN // 2, IMAGE_CHANNEL, 4, 2, 1, bias=False),
        nn.Tanh()
      )

### Discriminator Model:
      self.main = nn.Sequential(
        # 1st layer
        nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # 2nd layer
        nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(D_HIDDEN * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # 3rd layer
        nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(D_HIDDEN * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # 4th layer
        nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(D_HIDDEN * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # Adaptive Average Pooling to reduce to 1x1
        nn.AdaptiveAvgPool2d(1),
        # Output layer
        nn.Conv2d(D_HIDDEN * 8, 1, 1, 1, 0, bias=False),
        nn.Sigmoid()
    )

### Loss Function
    # Initialize BCELoss function
    adversarial_loss = nn.BCELoss()
    l1_loss = nn.L1Loss() 
    
    def criterion(output, label, real, fake):
        return adversarial_loss(output, label) #+ l1_loss(real, fake) 
    
    
    # Create batch of latent vectors that I will use to visualize the progression of the generator
    viz_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))



# Result
![image](https://github.com/user-attachments/assets/76eefd75-c5c6-4575-812a-d41932bc4671)

![image](https://github.com/user-attachments/assets/aa763409-5f57-42c9-864e-7b6e4e380fe6)
![image](https://github.com/user-attachments/assets/5917dd15-268f-493c-811b-543a61d4d32b)
![image](https://github.com/user-attachments/assets/d7832d6b-4baa-4eab-9adc-34e7acd71ef8)
![image](https://github.com/user-attachments/assets/3b572201-3adb-431e-9231-d8517327f68f)
![image](https://github.com/user-attachments/assets/a8b995b7-144e-4549-bbc0-ed8280523ad2)

    
