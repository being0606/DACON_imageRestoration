import os
import random
import zipfile
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist

# Suppress warnings for better log readability
warnings.filterwarnings("ignore")

# Configuration
CFG = {
    'EPOCHS': 30,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 32,
    'SEED': 42,
    'WORLD_SIZE': torch.cuda.device_count(),  # Number of GPUs available
    'NUM_WORKERS': 4,
    'PATIENCE': 5
}

# Seed setting
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can have performance implications
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def main():
    # Spawn processes
    mp.spawn(train, args=(CFG['WORLD_SIZE'],), nprocs=CFG['WORLD_SIZE'], join=True)

def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Seed everything
    seed_everything(CFG['SEED'] + rank)  # Ensure each process has a different seed

    # Create dataset and split into train and validation sets
    full_dataset = CustomDataset(damage_dir=damage_dir, origin_dir=origin_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG['SEED'])
    )

    # Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CFG['BATCH_SIZE'], sampler=train_sampler, num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CFG['BATCH_SIZE'], sampler=val_sampler, num_workers=CFG['NUM_WORKERS'], pin_memory=True
    )

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
                layers = [
                    nn.ConvTranspose2d(in_feat, out_feat, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_feat),
                    nn.ReLU(inplace=True)
                ]
                if dropout:
                    layers.append(nn.Dropout(dropout))
                return nn.Sequential(*layers)

            # Define the layers of the generator
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
            d1 = self.down1(x)  # 64
            d2 = self.down2(d1)  # 128
            d3 = self.down3(d2)  # 256
            d4 = self.down4(d3)  # 512
            d5 = self.down5(d4)  # 512
            d6 = self.down6(d5)  # 512
            d7 = self.down7(d6)  # 512
            d8 = self.down8(d7)  # 512

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
                layers = [
                    nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
                ]
                if normalization:
                    layers.append(nn.BatchNorm2d(out_filters))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return nn.Sequential(*layers)

            # Define the layers of the discriminator
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
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Wrap models with DistributedDataParallel
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[rank])
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])

    criterion_GAN = nn.MSELoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    optimizer_G = optim.Adam(
        generator.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        discriminator.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.5, 0.999)
    )

    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
    )
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_D, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
    )

    # Early stopping parameters
    early_stopping_patience = CFG['PATIENCE']
    epochs_without_improvement = 0

    # Mixed precision scaler
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # Training loop with validation
    best_val_loss = float("inf")
    lambda_pixel = 100

    for epoch in range(1, CFG['EPOCHS'] + 1):
        if epochs_without_improvement >= early_stopping_patience:
            if rank == 0:
                print("Early stopping triggered.")
            break

        # Set samplers epoch for shuffling
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)  # Not strictly necessary for val_sampler

        generator.train()
        discriminator.train()
        total_G_loss = 0
        total_D_loss = 0

        if rank == 0:
            print(f"\nEpoch [{epoch}/{CFG['EPOCHS']}]")

        train_bar = tqdm(train_loader, desc=f"Training Rank {rank}", leave=False, disable=(rank != 0))

        for i, batch in enumerate(train_bar):
            real_A = batch['A'].to(device, non_blocking=True)
            real_B = batch['B'].to(device, non_blocking=True)

            # ------------------
            #  Train Generator
            # ------------------
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast():
                fake_B = generator(real_A)
                pred_fake = discriminator(fake_B, real_A)
                valid = torch.ones_like(pred_fake, device=device, requires_grad=False)
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
            with torch.cuda.amp.autocast():
                pred_real = discriminator(real_B, real_A)
                valid = torch.ones_like(pred_real, device=device, requires_grad=False)
                loss_real = criterion_GAN(pred_real, valid)

                pred_fake = discriminator(fake_B.detach(), real_A)
                fake = torch.zeros_like(pred_fake, device=device, requires_grad=False)
                loss_fake = criterion_GAN(pred_fake, fake)

                loss_D = 0.5 * (loss_real + loss_fake)

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            total_G_loss += loss_G.item()
            total_D_loss += loss_D.item()

            if rank == 0:
                # Update progress bar
                train_bar.set_postfix({
                    'Batch': f'{i}/{len(train_loader)}',
                    'D_loss': f'{loss_D.item():.4f}',
                    'G_loss': f'{loss_G.item():.4f}'
                })

        # Average the losses across all processes
        total_G_loss_tensor = torch.tensor(total_G_loss, device=device)
        total_D_loss_tensor = torch.tensor(total_D_loss, device=device)

        dist.all_reduce(total_G_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_D_loss_tensor, op=dist.ReduceOp.SUM)

        avg_G_loss = total_G_loss_tensor.item() / (len(train_loader) * world_size)
        avg_D_loss = total_D_loss_tensor.item() / (len(train_loader) * world_size)

        # Gather validation loss from all processes
        generator.eval()
        val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Validation Rank {rank}", leave=False, disable=(rank != 0))
            for batch in val_bar:
                real_A = batch['A'].to(device, non_blocking=True)
                real_B = batch['B'].to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    fake_B = generator(real_A)
                    loss_pixel = criterion_pixelwise(fake_B, real_B)
                val_loss += loss_pixel.item()

        # Reduce val_loss across all processes
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.reduce(val_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
        total_val_loss = val_loss_tensor.item()

        if rank == 0:
            val_loss_avg = total_val_loss / (len(val_loader) * world_size)
            print(f"Epoch [{epoch}/{CFG['EPOCHS']}], "
                  f"Generator Loss: {avg_G_loss:.4f}, Discriminator Loss: {avg_D_loss:.4f}")
            print(f"Validation Loss: {val_loss_avg:.4f}")

            # Step schedulers
            scheduler_G.step(val_loss_avg)
            scheduler_D.step(val_loss_avg)

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_without_improvement = 0
                os.makedirs("./saved_models", exist_ok=True)
                torch.save(generator.module.state_dict(), "./saved_models/best_generator.pth")
                torch.save(discriminator.module.state_dict(), "./saved_models/best_discriminator.pth")
                print(f"Best model saved with validation loss: {best_val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs.")

    # Cleanup
    dist.destroy_process_group()

    if rank == 0:
        # Only the main process performs testing and saving results
        test_and_save_results(generator.module, device)

def test_and_save_results(generator, device):
    # Testing and saving results
    submission_dir = "./data/submission"
    os.makedirs(submission_dir, exist_ok=True)

    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = transforms.Resize((256, 256))(image)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5]*3, [0.5]*3)(image)
        image = image.unsqueeze(0)
        return image

    generator.eval()

    test_images = sorted(os.listdir(test_dir))

    # Progress bar for testing
    test_bar = tqdm(test_images, desc="Testing", leave=False)

    for image_name in test_bar:
        test_image_path = os.path.join(test_dir, image_name)
        test_image = load_image(test_image_path).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred_image = generator(test_image)
            pred_image = pred_image.cpu().squeeze(0)
            pred_image = pred_image * 0.5 + 0.5  # Denormalize
            pred_image = pred_image.numpy().transpose(1, 2, 0)
            pred_image = (pred_image * 255).astype('uint8')
            pred_image_resized = cv2.resize(
                pred_image, (512, 512), interpolation=cv2.INTER_LINEAR
            )

        output_path = os.path.join(submission_dir, image_name)
        cv2.imwrite(output_path, cv2.cvtColor(pred_image_resized, cv2.COLOR_RGB2BGR))

    print(f"Saved all images to {submission_dir}")

    # Create submission ZIP file
    zip_filename = "./data/submission_DDP.zip"
    with zipfile.ZipFile(zip_filename, 'w') as submission_zip:
        for image_name in test_images:
            image_path = os.path.join(submission_dir, image_name)
            submission_zip.write(image_path, arcname=image_name)

    print(f"All images saved in {zip_filename}")

if __name__ == '__main__':
    main()