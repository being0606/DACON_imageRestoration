#!/usr/bin/env python
# coding: utf-8

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import lightning as L
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from glob import glob
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

# 하이퍼파라미터 설정
SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.225, 0.225, 0.225]

TRAIN_DATA_DIR = "./data/train_gt"
TEST_DATA_DIR = "./data/test_input"
MASK_DIR = "./data/train_masks"
SUBMISSION_DATA_DIR = "./submission"
EXPERIMENT_NAME = "baseline"

# 시드 고정
L.seed_everything(SEED)


# 1. 마스크 생성 함수 정의
def create_masks(image_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for image_path in tqdm(glob(f"{image_dir}/*.png"), desc="마스크 생성 중"):
        image = Image.open(image_path)
        width, height = image.size
        mask = Image.new("L", (width, height), color=0)
        draw = ImageDraw.Draw(mask)
        rect_width = width // 4
        rect_height = height // 4
        rect_x1 = random.randint(0, width - rect_width)
        rect_y1 = random.randint(0, height - rect_height)
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=255)
        mask_name = os.path.basename(image_path).replace(".png", "_mask.png")
        mask_path = os.path.join(mask_dir, mask_name)
        mask.save(mask_path)
    print(f"모든 마스크 생성 완료: {mask_dir}")


# 2. 데이터 전처리 및 CSV 파일 생성
def create_csv_files():
    train_image_paths = sorted(glob(f"{TRAIN_DATA_DIR}/*.png"))
    train_df = pd.DataFrame(
        {
            "image": [os.path.basename(path) for path in train_image_paths],
            "mask": [
                os.path.basename(path).replace(".png", "_mask.png")
                for path in train_image_paths
            ],
        }
    )
    train_df.to_csv("./preproc/train_preproc.csv", index=False)
    create_masks(TRAIN_DATA_DIR, MASK_DIR)
    test_image_paths = sorted(glob(f"{TEST_DATA_DIR}/*.png"))
    test_df = pd.DataFrame(
        {"image": [os.path.basename(path) for path in test_image_paths]}
    )
    test_df.to_csv("./preproc/test_preproc.csv", index=False)
    print("train_preproc.csv 및 test_preproc.csv 생성 완료")


if not os.path.exists("./preproc/train_preproc.csv") or not os.path.exists(
    "./preproc/test_preproc.csv"
):
    create_csv_files()

train_df = pd.read_csv("./preproc/train_preproc.csv")
test_df = pd.read_csv("./preproc/test_preproc.csv")


# 3. 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, mask_dir=None, mode="train"):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.mask_dir = mask_dir
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        if self.mode == "train":
            mask_path = os.path.join(self.mask_dir, self.df.iloc[idx, 1])
            mask = Image.open(mask_path).convert("L")
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, torch.tensor(np.array(mask), dtype=torch.long)
        else:
            return self.transform(image) if self.transform else image


# 4. 이미지 전처리 정의
image_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]
)

train_dataset = CustomImageDataset(
    train_df,
    data_dir=TRAIN_DATA_DIR,
    transform=image_transform,
    mask_dir=MASK_DIR,
    mode="train",
)
test_dataset = CustomImageDataset(
    test_df, data_dir=TEST_DATA_DIR, transform=image_transform, mode="test"
)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 5. 모델 정의 및 LightningModule 구현
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
)

lit_model = LitModel(model=model)


# 6. 모델 학습 (단일 GPU 환경)
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    mode="min",
    dirpath="./checkpoints",
    filename="baseline-{epoch:02d}-{train_loss:.2f}",
    save_top_k=1,
    verbose=True,
)

trainer = L.Trainer(
    max_epochs=EPOCHS,
    precision="bf16-mixed",
    callbacks=[checkpoint_callback],
    log_every_n_steps=10,
    devices=1,  # 단일 GPU
    accelerator="gpu",
)

trainer.fit(lit_model, train_dataloader)


# 7. 테스트 및 제출 파일 생성
os.makedirs(SUBMISSION_DATA_DIR, exist_ok=True)

predictions = []
for images in tqdm(test_dataloader, desc="테스트 진행 중"):
    images = images.to("cuda")
    with torch.no_grad():
        preds = lit_model(images).argmax(dim=1).cpu().numpy()
        predictions.extend(preds)

submission_file = os.path.join(SUBMISSION_DATA_DIR, f"{EXPERIMENT_NAME}.csv")
test_df["label"] = predictions
test_df.to_csv(submission_file, index=False)

print(f"제출 파일 생성 완료: {submission_file}")
