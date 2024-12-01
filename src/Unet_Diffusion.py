import os
import random
import zipfile

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusionInpaintPipeline  # Pre-trained Diffusion Inpainting Model

# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Configuration
CFG = {
    'EPOCHS': 30,
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'SEED': 42,
    'PATIENCE': 10
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
class ImageDataset(Dataset):
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
        origin_img = Image.open(origin_img_path).convert("L")  # 그레이스케일로 변경

        if self.transform:
            # Apply the same transformation to both images
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            damage_img = self.transform(damage_img)
            random.seed(seed)
            torch.manual_seed(seed)
            origin_img = self.transform(origin_img)
        else:
            damage_img = transforms.ToTensor()(damage_img)
            origin_img = transforms.ToTensor()(origin_img)

        return {'A': damage_img, 'B': origin_img}

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create dataset and split into train and validation
full_dataset = ImageDataset(damage_dir=damage_dir, origin_dir=origin_dir, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(CFG['SEED'])
)

# Adjust num_workers according to system cores
import multiprocessing
num_workers = multiprocessing.cpu_count()

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CFG['BATCH_SIZE'],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# Define UNet model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.down1 = conv_block(in_channels, 64, normalize=False)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.center = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)
        c = self.center(p4)
        u4 = self.up4(c)
        m4 = torch.cat([u4, d4], dim=1)
        d4 = self.dec4(m4)
        u3 = self.up3(d4)
        m3 = torch.cat([u3, d3], dim=1)
        d3 = self.dec3(m3)
        u2 = self.up2(d3)
        m2 = torch.cat([u2, d2], dim=1)
        d2 = self.dec2(m2)
        u1 = self.up1(d2)
        m1 = torch.cat([u1, d1], dim=1)
        d1 = self.dec1(m1)
        out = self.final(d1)
        return out

# Initialize models, optimizers, and loss functions
unet = UNet().to(device)

# Use DataParallel for multi-GPU support
if torch.cuda.device_count() > 1:
    unet = nn.DataParallel(unet, device_ids=[0, 1])

# Loss function
mask_loss = nn.BCEWithLogitsLoss()

# Optimizer for U-Net
optimizer_unet = optim.Adam(unet.parameters(), lr=CFG['LEARNING_RATE'], betas=(0.5, 0.999))

# Learning Rate Scheduler
scheduler_unet = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, mode='min', factor=0.5, patience=5)

# Load pre-trained diffusion model
diffusion_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting"
).to(device)

epochs = CFG['EPOCHS']
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)
checkpoint_path = "checkpoint.pth"

best_val_loss = float("inf")
early_stopping_counter = 0
scaler_unet = torch.cuda.amp.GradScaler()

# Training loop for U-Net
for epoch in range(epochs):
    unet.train()
    running_loss_unet = 0.0

    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for batch in train_loader:
            input_images = batch['A'].to(device)
            gt_images = batch['B'].to(device)

            # Train U-Net for mask detection
            optimizer_unet.zero_grad()
            with torch.cuda.amp.autocast():
                predicted_masks = unet(input_images)
                loss_unet = mask_loss(predicted_masks, (gt_images > 0).float())

            scaler_unet.scale(loss_unet).backward()
            scaler_unet.step(optimizer_unet)
            scaler_unet.update()

            running_loss_unet += loss_unet.item()

            pbar.set_postfix(unet_loss=loss_unet.item())
            pbar.update(1)

    scheduler_unet.step(running_loss_unet / len(train_loader))
    epoch_loss = running_loss_unet / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - U-Net Loss: {epoch_loss:.4f}")

    # Validation
    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        with tqdm(total=len(val_loader), desc="Validation", unit="batch") as val_pbar:
            for batch in val_loader:
                input_images = batch['A'].to(device)
                gt_images = batch['B'].to(device)
                predicted_masks = unet(input_images)
                loss_unet = mask_loss(predicted_masks, (gt_images > 0).float())
                val_loss += loss_unet.item()
                val_pbar.set_postfix(unet_loss=loss_unet.item())
                val_pbar.update(1)

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save the model with the lowest validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0
        torch.save({
            'epoch': epoch,
            'unet_state_dict': unet.state_dict(),
            'optimizer_unet_state_dict': optimizer_unet.state_dict()
        }, checkpoint_path)
        print(f"Best U-Net model saved with validation loss: {best_val_loss:.4f}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= CFG['PATIENCE']:
            print("Early stopping triggered")
            break

# Testing and saving results
submission_dir = "./data/submission"
os.makedirs(submission_dir, exist_ok=True)
unet.eval()

test_images = sorted(os.listdir(test_dir))

with torch.no_grad():
    with tqdm(total=len(test_images), desc="Testing", unit="image") as test_pbar:
        for img_name in test_images:
            input_path = os.path.join(test_dir, img_name)
            input_image = Image.open(input_path).convert("RGB")
            input_tensor = test_transform(input_image).unsqueeze(0).to(device)
            predicted_mask = unet(input_tensor).squeeze().cpu().numpy()
            predicted_mask = (predicted_mask > 0.5).astype(np.uint8) * 255
            mask_image = Image.fromarray(predicted_mask).convert("L")

            # Use the pre-trained diffusion model for inpainting
            inpainted_image = diffusion_pipeline(
                prompt="A high-quality realistic image",
                image=input_image,
                mask_image=mask_image
            ).images[0]

            output_path = os.path.join(submission_dir, img_name)
            inpainted_image.save(output_path)

            test_pbar.update(1)

print(f"Saved all images to {submission_dir}")

# Create submission ZIP file
zip_filename = "./data/submission.zip"
with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
    for img_name in test_images:
        img_path = os.path.join(submission_dir, img_name)
        submission_zip.write(img_path, arcname=img_name)

print(f"All images saved in {zip_filename}")
