import os
import random
import zipfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Configuration
CFG = {
    'EPOCHS': 10,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 16,
    'SEED': 42
}

# Seed setting
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG['SEED'])

# Data directories
origin_dir = './data/train_gt'
damage_dir = './data/train_input'
test_dir = './data/test_input'

# Custom dataset with consistent transformations
class CustomDataset(Dataset):
    def __init__(self, damage_dir, origin_dir, transform=None):
        self.damage_dir = damage_dir
        self.origin_dir = origin_dir
        self.transform = transform
        self.damage_files = sorted(os.listdir(damage_dir))
        self.origin_files = sorted(os.listdir(origin_dir))

        assert len(self.damage_files) == len(self.origin_files), \
            "The number of images in damage and origin folders must match"

    def __len__(self):
        return len(self.damage_files)

    def __getitem__(self, idx):
        damage_img_name = self.damage_files[idx]
        origin_img_name = self.origin_files[idx]

        damage_img_path = os.path.join(self.damage_dir, damage_img_name)
        origin_img_path = os.path.join(self.origin_dir, origin_img_name)

        damage_img = Image.open(damage_img_path).convert("RGB")
        origin_img = Image.open(origin_img_path).convert("RGB")

        if self.transform:
            # Apply the same transformation to both images
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            damage_img = self.transform(damage_img)
            random.seed(seed)
            torch.manual_seed(seed)
            origin_img = self.transform(origin_img)

        return {'A': damage_img, 'B': origin_img}

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create dataset and split into train and validation
full_dataset = CustomDataset(damage_dir=damage_dir, origin_dir=origin_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(CFG['SEED']))

train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=4)

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()

        def down_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_feat, out_feat, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                      nn.BatchNorm2d(out_feat),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, 64, normalize=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)
        self.down6 = down_block(512, 512)
        self.down7 = down_block(512, 512)
        self.down8 = down_block(512, 512, normalize=False)

        self.up1 = up_block(512, 512, dropout=0.5)
        self.up2 = up_block(1024, 512, dropout=0.5)
        self.up3 = up_block(1024, 512, dropout=0.5)
        self.up4 = up_block(1024, 512)
        self.up5 = up_block(1024, 256)
        self.up6 = up_block(512, 128)
        self.up7 = up_block(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))

        return u8

# PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels * 2, 64, normalization=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# Initialize models, optimizers, and loss functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.5, 0.999))

# Update: Use torch.amp instead of torch.cuda.amp
scaler_G = torch.amp.GradScaler()
scaler_D = torch.amp.GradScaler()

# Training loop with validation and tqdm progress bars
best_val_loss = float("inf")
lambda_pixel = 100

for epoch in range(1, CFG['EPOCHS'] + 1):
    generator.train()
    discriminator.train()
    total_G_loss = 0
    total_D_loss = 0

    print(f"\nEpoch [{epoch}/{CFG['EPOCHS']}]")
    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for i, batch in enumerate(train_bar):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            valid = torch.ones_like(pred_fake, device=device)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_pixel = criterion_pixelwise(fake_B, real_B)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

        scaler_G.scale(loss_G).backward()
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            pred_real = discriminator(real_B, real_A)
            valid = torch.ones_like(pred_real, device=device)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(fake_B.detach(), real_A)
            fake = torch.zeros_like(pred_fake, device=device)
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = 0.5 * (loss_real + loss_fake)

        scaler_D.scale(loss_D).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()

        # Update progress bar
        train_bar.set_postfix({
            'Batch': f'{i}/{len(train_loader)}',
            'D_loss': f'{loss_D.item():.4f}',
            'G_loss': f'{loss_G.item():.4f}'
        })

    # Step-by-step logging after each epoch
    avg_G_loss = total_G_loss / len(train_loader)
    avg_D_loss = total_D_loss / len(train_loader)
    print(f"Epoch [{epoch}/{CFG['EPOCHS']}], Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}")

    # Validation
    generator.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in val_bar:
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            with torch.amp.autocast(device_type='cuda'):
                fake_B = generator(real_A)
                loss_pixel = criterion_pixelwise(fake_B, real_B)
            val_loss += loss_pixel.item()

            # Update progress bar
            val_bar.set_postfix({'Pixel_Loss': f'{loss_pixel.item():.4f}'})

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Save the model with the lowest validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("./saved_models", exist_ok=True)
        torch.save(generator.state_dict(), "./saved_models/best_generator.pth")
        torch.save(discriminator.state_dict(), "./saved_models/best_discriminator.pth")
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

# Testing and saving results
submission_dir = "../data/submission"
os.makedirs(submission_dir, exist_ok=True)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

generator.load_state_dict(torch.load("./saved_models/best_generator.pth"))
generator.eval()

test_images = sorted(os.listdir(test_dir))

# Progress bar for testing
test_bar = tqdm(test_images, desc="Testing", leave=False)

for image_name in test_bar:
    test_image_path = os.path.join(test_dir, image_name)
    test_image = load_image(test_image_path).to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            pred_image = generator(test_image)
        pred_image = pred_image.cpu().squeeze(0)
        pred_image = pred_image * 0.5 + 0.5
        pred_image = pred_image.numpy().transpose(1, 2, 0)
        pred_image = (pred_image * 255).astype('uint8')
        pred_image_resized = cv2.resize(pred_image, (512, 512), interpolation=cv2.INTER_LINEAR)

    output_path = os.path.join(submission_dir, image_name)
    cv2.imwrite(output_path, cv2.cvtColor(pred_image_resized, cv2.COLOR_RGB2BGR))

print(f"Saved all images to {submission_dir}")

# Create submission ZIP file
zip_filename = "./data/submission.zip"
with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
    for image_name in test_images:
        image_path = os.path.join(submission_dir, image_name)
        submission_zip.write(image_path, arcname=image_name)

print(f"All images saved in {zip_filename}")