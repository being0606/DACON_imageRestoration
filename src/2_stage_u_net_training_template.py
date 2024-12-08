#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import segmentation_models_pytorch as smp

from tqdm.auto import tqdm
from glob import glob
from PIL import Image
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as ski_ssim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

# 하이퍼파라미터 설정
SEED = 42
BATCH_SIZE = 16
N_SPLIT = 5
IMAGE_PREPROC_MEAN = 0.5
IMAGE_PREPROC_STD = 0.225
MIN_POLYGON_BBOX_SIZE = 50

TRAIN_DATA_DIR = "./data/train_gt"
VALID_DATA_DIR = "./data/valid_input"
TEST_DATA_DIR = "./data/test_input"
SUBMISSION_DATA_DIR = "./submission"
EXPERIMENT_NAME = "baseline"

L.seed_everything(SEED)

# 데이터 로드
train_df = pd.read_csv("./preproc/train_preproc.csv")
test_df = pd.read_csv("./preproc/test_preproc.csv")


# 데이터셋 클래스
class CustomImageDataset(Dataset):
    def __init__(
        self,
        df,
        data_dir="./data/train_gt",
        mode="train",
        min_polygon_bbox_size=MIN_POLYGON_BBOX_SIZE,
    ):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.min_polygon_bbox_size = min_polygon_bbox_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx, 0])
        if self.mode == "train":
            return get_input_image(Image.open(img_path), self.min_polygon_bbox_size)
        elif self.mode == "valid":
            return self.load_input_image(img_path)
        elif self.mode == "test":
            return {"image_gray_masked": Image.open(img_path)}

    def load_input_image(self, img_input_path):
        return np.load(img_input_path, allow_pickle=True).item()


# 모델 클래스 정의
class LitIRModel(L.LightningModule):
    def __init__(self, model_1, model_2):
        super().__init__()
        self.model_1 = model_1
        self.model_2 = model_2

    def forward(self, images_gray_masked):
        images_gray_restored = self.model_1(images_gray_masked) + images_gray_masked
        images_restored = self.model_2(images_gray_restored)
        return images_gray_restored, images_restored

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)


# 모델 초기화
model_1 = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=1
)
model_2 = smp.Unet(
    encoder_name="resnet34", encoder_weights="imagenet", in_channels=1, classes=3
)
lit_ir_model = LitIRModel(model_1=model_1, model_2=model_2)

# 데이터로더 설정
train_dataset = CustomImageDataset(train_df, data_dir=TRAIN_DATA_DIR, mode="train")
valid_dataset = CustomImageDataset(train_df, data_dir=VALID_DATA_DIR, mode="valid")
test_dataset = CustomImageDataset(test_df, data_dir=TEST_DATA_DIR, mode="test")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

# 모델 학습
trainer = L.Trainer(
    max_epochs=100,
    precision="bf16-mixed",
    callbacks=[
        ModelCheckpoint(monitor="val_score", mode="max", dirpath="./checkpoint/")
    ],
)
trainer.fit(lit_ir_model, train_dataloader, valid_dataloader)

# 예측
predictions = trainer.predict(lit_ir_model, test_dataloader)

# 제출 파일 생성
os.makedirs(SUBMISSION_DATA_DIR, exist_ok=True)
for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    image_pred = Image.fromarray(predictions[idx])
    image_pred.save(os.path.join(SUBMISSION_DATA_DIR, row["image"]), "PNG")
