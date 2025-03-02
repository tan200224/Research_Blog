# Condtional GAN for medical image

## Data Processing:
Convert all the 4x512x512 into 1x256x256

    class CTScan(Dataset):
        def __init__(self, data_dir, transform=None):
            self.dir = data_dir
            self.img_files = sorted(glob(os.path.join(self.dir, 'train', '*nii.gz')))
            self.label_files = sorted(glob(os.path.join(self.dir, 'label', '*nii.gz')))
            self.transform = transform
            self.slices = []
    
            # Calculate the total number of slices
            for img_path, label_path in zip(self.img_files, self.label_files):
                img_nii = nib.load(img_path)
                num_slices = img_nii.shape[2]
                for s in range(num_slices):
                    self.slices.append((img_path, label_path, s))
    
        def __len__(self):
            return len(self.slices)
    
        def __getitem__(self, idx):
            img_path, label_path, slice_idx = self.slices[idx]
    
            # Load the image and label
            img_nii = nib.load(img_path)
            label_nii = nib.load(label_path)
    
            # Extract the specific slice
            img_slice = img_nii.get_fdata()[:, :, slice_idx]
            label_slice = label_nii.get_fdata()[:, :, slice_idx]
    
            # Convert to tensor
            img_slice = torch.tensor(img_slice, dtype=torch.float32)
            label_slice = torch.tensor(label_slice, dtype=torch.float32)
    
            # Normalize the image slice
            img_slice = np.clip(img_slice, -50, 200)
            min_val = img_slice.min()
            max_val = img_slice.max()
            img_slice = (img_slice - min_val) / (max_val - min_val)
    
            # Add a channel dimension (PyTorch expects CxHxW)
            img_slice = img_slice.unsqueeze(0)
            label_slice = label_slice.unsqueeze(0)
    
            # Resize the image and label
            img_slice = TF.resize(img_slice, size=(256, 256))
            label_slice = TF.resize(label_slice, size=(64, 64))
    
            return {"data": img_slice, "label": label_slice}


# Generator

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # Input layer: Upsample from Z_DIM to 4x4 spatial resolution
                nn.ConvTranspose2d(4196, G_HIDDEN * 16, 4, 1, 0, bias=False),
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

    def forward(self, input):
        return self.main(input)  

# Discriminator 

    class Discriminator(nn.Module):
        def __init__(self, in_channels=1):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                # 1st layer
                nn.Conv2d(in_channels, D_HIDDEN, 4, 2, 1, bias=False),
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
    
        def forward(self, input):
            return self.main(input).view(-1, 1).squeeze(1)



# Training 

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    best_loss = np.inf
    
    print("Starting Training Loop...")
    for epoch in range(EPOCH_NUM):
        for i, data in enumerate(train_loader):
            
            
            # (1) Update the discriminator with real data
            netD.zero_grad()
            # Format batch
            real = data['data'].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
    
            # Forward pass real batch through D
            output = netD(real).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
    
            # (2) Update the discriminator with fake data
            # Generate batch of latent vectors
            label_data = data['label'].to(device)  # Shape [batch_size, 1, 64, 64]
            
            label_data = label_data.view(b_size, -1)  # Reshape to [batch_size, 4096]
            linear_transform = nn.Linear(4096, 4096 * 1 * 1).to(device)  # Adjust output size to match noise dimensions
            label_data = linear_transform(label_data).view(b_size, 4096, 1, 1)  # Reshape to [batch_size, Z_DIM, 1, 1]
    
            # Generate noise and concatenate with label data
            noise = torch.randn(b_size, 100, 1, 1, device=device)
            noise = torch.cat((label_data, noise), dim=1)  # Concatenate along the channel dimension
            
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(FAKE_LABEL)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            if i % 4 == 0:
                optimizerD.step()
    
            # (3) Update the generator with fake data
            netG.zero_grad()
            label.fill_(REAL_LABEL)  # fake labels are real for generator cost
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
    
            # Output training stats
            if epoch % 100 == 0 and iters % 10000 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, EPOCH_NUM, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % 100 == 0 and iters % 10000 == 0):
                with torch.no_grad():
                    fake = netG(noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, normalize=True))
    
                img = fake.squeeze(0)  # This removes the batch dimension, resulting in shape [256, 256, 1]
                img = img.permute(2, 0, 1) 
                plt.imshow(img[:, 0, :], cmap='gray')
                plt.show()
    
            iters += 1
        
        # Show the image after every 100 epoch
    #     if (epoch % 100 == 0):
    #         img = img_list[-1].permute(1, 2, 0)  # Assuming img_list is a tensor, permute to (H, W, C)
    #         f, axarr = plt.subplots(1, 3)
    #         axarr[0].imshow(img[:, :, 0], cmap='gray')
    #         axarr[1].imshow(img[:, :, 1], cmap='gray')
    #         axarr[2].imshow(img[:, :, 2], cmap='gray')
    #         plt.show()
        
        # Update the best loss after each epoch
        if epoch % 500 == 0:
            best_loss = errG + errD
            print("Best Loss: " + str(best_loss) + " at " + str(epoch))
            torch.save({
                'epoch': epoch,
                'Generator_state_dict': netG.state_dict(),
                'Discriminator_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'errG': errG,
                'errD': errD,
                }, r'D:..\..\DataBase\combine\GAN Model\GANImgToImg_4_tmp.pth')


# Result for 25 slices

![Screenshot 2024-11-27 010215](https://github.com/user-attachments/assets/b4b0277e-1a57-4242-990f-aebddd95c3e9)
![Screenshot 2024-11-27 010222](https://github.com/user-attachments/assets/0569b501-ff63-4d09-b104-7f42ce305793)
![Screenshot 2024-11-27 010207](https://github.com/user-attachments/assets/bb39ed38-2669-4221-b31d-487e8d2c982c)












