---
title:  "[Kaggle] BYU - Locating Bacterial Flagellar Motors 2025"
header:
   teaser: "/assets/images/BYU images/header.png" 
excerpt: "캐글 대회에서 실패를 통해 배운 교훈"
categories: 
- AI
- Detection
tags:
- competition
toc_label: Contents
toc: true
toc_sticky: True
toc_h_min: 1
toc_h_max: 3
date: 2025-09-04
last_modified_at: 2025-09-04
---

# About Competition
## Overview

![image](/assets/images/BYU images/header.png){: width="100%" height="100%"}

[BYU - Locating Bacterial Flagellar Motors 2025](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025)


**대회 기간**: 2025.03.06 ~ 2025.06.04

**문제 및 목표**: 3차원의 박테리아를 단층 촬영한 이미지들(tomogram)이 있을 때, 박테리아의 모터 좌표(x,y,z)를 탐지한다.

**박테리아의 모터란?**: 박테리아는 이동하기 위한 편모(꼬리처럼 생김)가 있고 편모와 몸체가 연결된 부분이 모터다. 박테리아는 지구상에서 유일하게 원운동(바퀴처럼 완전히 회전)하는 기관(모터)을 가진 유기물이다.

![image](/assets/images/BYU images/tomogram.gif){: width="100%" height="100%"}

위 영상은 단층 촬영한 이미지들을 쌓아서 영상으로 만든 것 입니다. 짧게 빨간 점이 점멸하는 부분이 라벨링된 모터의 좌표(x,y,z)입니다.

***

이번 캐글 대회에서 수상하지는 못했지만 과정을 통해 배운점이 많습니다. 실패를 통해 배운 교훈을 설명하겠습니다.

## Dataset
![image](/assets/images/BYU images/Dataset.png){: width="100%" height="100%"}

학습 데이터, 검증 데이터 각각 단층 촬영된 박테리아의 폴더(e.g. tomo_00e047)가 있고 각 폴더 안에는 단층 촬영된 이미지(e.g. slice_0000.jpg)들이 있습니다.\
학습 데이터에는 tomogram이 737개 존재하며, 한 tomogram 당 300~600개의 이미지(slice)가 존재합니다.

### train_labels.csv
![image](/assets/images/BYU images/train_labels.png){: width="100%" height="100%"}

**train_labels.csv의 컬럼 요약** : `tomogram의 고유  id` , `z, y, x 좌표` , `이미지의 z`, `height, width 크기` , `이미지의 한 픽셀 당 담긴angstroms(원자 하나의 지름과 비슷한 거리 단위) 크기` , `한 tomogram에 있는 모터의 개수`

test 데이터에는 더미로 학습 데이터에 있는 tomogram 데이터가 3개 들어있고, 제출했을 때 공개되지 않은 900개의 tomogram으로 점수가 평가됩니다. **실제 test 데이터에는 모터가 없거나 모터가 딱 하나인 데이터들만 존재**합니다.

### Evaluation
![image](/assets/images/BYU images/metric.png){: width="100%" height="100%"}

예측한 모터의 좌표가 정답 좌표와 1000 angstroms 내에 있으면 TP, 밖에 있으면 FN으로 개수를 셉니다. 그리고 $F_\beta$-score로 평가합니다. 


# Workflow
## 1. 빠른 제출 
![image](/assets/images/BYU images/baseline code.png){: width="100%" height="100%"}

다른 참여자가 작성한 **baseline** 코드를 복사해서 **데이터 생성, 모델 학습, 제출** 과정까지 빠르게 거쳤습니다. 그랬을 때 **public 점수**는 **0.507**이었습니다. 당시 **1등**의 점수는 **0.88**였습니다.

해당 코드에서 이미지가 가지고 있는 값의 상위 2%, 하위 2%를 제거하고 제거한만큼 중간 값의 범위를 늘리는 방식을 사용했습니다. 이를 통해 흐릿한 이미지의 대비를 더 극명하게 할 수 있음으로 저도 제 코드에 적극 활용했습니다.


```python
# baseline에 있는 코드
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    
    return np.uint8(normalized)

# 내가 수정한 torch 버전
def contrast_extension(self, data):
    p2 = torch.quantile(data, 0.02)
    p98 = torch.quantile(data, 0.98)
    
    # 클리핑
    clipped_data = torch.clamp(data, min=p2, max=p98)
    
    # 0~1 범위로 규제
    return (clipped_data - p2) / (p98 - p2)
```

## 2. 모델선정

일반적으로 decetion에서 사용하는 모델이 yolo임으로 저도 **yolo를 채택**했습니다.\
모델의 아키텍처를 개조하고 싶어 처음에는 구조가 간단한 **yolo8**를 사용하였고, 나중에는 **yolo11**도 사용해 성능을 확인했습니다. 

## 3. Overlay로 데이터 생성

yolo는 태생적으로 **채널이 3(RGB)이거나 1(gray)인 이미지만**을 입력으로 받을 수 있기 때문에 3차원 정보를 담은 tomogram을 어떻게 yolo의 입력으로 넣을 수 있을지 고민하였습니다. 첫 번째로 선택한 방법이 **오버레이(overlay)로 3차원 이미지들을 하나로 합쳐 2차원에 모두 담길 수 있도록 하는 것**이었습니다.\
모터의 위치를 추정하기 위해서는 모델이 **편모의 전체적인 형태**를 파악하는 것이 유리할 것입니다. 그런데 개별적인 2차원 이미지를 봤을 때 편모는 일부만 드러나고 끊겨서 보입니다. 그렇기에 3차원의 이미지들을 모델이 한 번에 볼 수 있도록 하는 것이 좋다고 생각했습니다.\
하지만 오버레이 방식은 결과적으로 **한계**가 있다고 판단하여 사용하지 않았습니다. 그렇게 판단하게된 과정을 설명하겠습니다.


```python
def overlay_blend(background, foreground):
    """Overlay 블렌딩 모드 적용"""
    result = torch.where(
        background < 0.5,
        2 * background * foreground,  # Multiply 적용
        1 - (2 * (1 - background) * (1 - foreground)) # -0.1* background**2 # Screen 적용
    )
    return torch.clip(result, 0, 1)
```

오버레이는 배경 이미지의 픽셀 값이 어두우면 더 어두워지도록 얹는 이미지의 픽셀 값과 **Multiply**하고, 밝으면 더 밝아지도록 얹는 이미지의 픽셀 값과 **Screen** 연산을 합니다.

하지만 오버레이에는 몇 가지 **문제**가 있었습니다. 

### 오버레이 첫 번째 문제
![image](/assets/images/BYU images/overlay1.png){: width="100%" height="100%"}

**적은 수의 이미지**가 오버레이 됐을 때는 **괜찮은 퀄리티**로 이미지가 **합쳐**지지만 **많이 쌓을 수록** 편모의 형태가 **소실**됩니다. 이를 해결하기 위해 오랜 시간을 들여 **수학식**을 개선해보려 했지만, 새로운 이미지를 쌓을수록 얹는 이미지의 **밝은 색(값)도 계속 쌓임**으로 Multiply를 한다고 해도 편모의 픽셀 값이 **결국엔 밝아져 형태가 희미**해지는 문제가 생깁니다.\
이를 수학적으로 해결하는 것에는 **한계**가 있다고 생각하여 **오버레이할 이미지만을 판별**할 수 있는 **binary classification** 모델을 만들기로 했습니다. 
오버레이할 이미지는 **박테리아의 존재 여부**로 판별됩니다. 모터(그리고 편모)는 없지만 박테리아의 몸통만 존재하는 tomogram이 있음으로 모터가 없는 tomogram과 있는 tomogram으로 분류 모델을 학습시키는 것이 아니라, 모터가 있는 tomogram의 **정답 z**에서 **근접 범위를 박테리아가 있는 이미지**, **멀리 떨어진 이미지를 박테리아가 없는 이미지**로 학습 데이터를 구성해서 모델을 학습시켰습니다. 

아래는 **binary classification** 모델을 학습시키는 코드입니다.


```python
import torch 
import torchvision.models as models
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import data 
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import ResNet34_Weights
from collections import Counter
import polars as pl
import os
import math as m
from tqdm.auto import tqdm 
from time import time
```


```python
train_val_rate = 0.9
batch_size = 64
epochs = 10
learning_rate = 1e-4
```


```python
class Classification_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir = './', split='train', validation=False, train_val_rate=0.9, base_transform=None, aug_transform=None): 
        super(Classification_Dataset, self).__init__() # 상속
        
        self.df = pl.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv", columns=["tomo_id","Motor axis 0","Voxel spacing"], low_memory=True)
        limit_angstroms = 1000 / 3**(1/2) # 박테리아가 있다고 예측할 데이터 경계(angstroms)
        outlier_angstroms = 2000 # 박테리아가 없다고 예측할 데이터 경계(angstroms)
        
        self.top_directory = os.path.join(root_dir, split) 

        self.data_list = []  # self.data_list에 모든 파일 목록 저장
        for sub_directory in os.listdir(self.top_directory):
            full_directory = os.path.join(self.top_directory, sub_directory)

            df_current_tomo = self.df.filter(pl.col("tomo_id")==sub_directory)
            voxel_spacing = df_current_tomo.select("Voxel spacing")[0].item() 
            limit_z = m.trunc(limit_angstroms / voxel_spacing)
            outlier_z = m.trunc(outlier_angstroms / voxel_spacing) 
            gt_max = int(df_current_tomo.select("Motor axis 0").max().item())
            gt_min = int(df_current_tomo.select("Motor axis 0").min().item())
            
            if gt_max != -1: # 모터가 있는 데이터로만 학습 데이터를 구성
                for file_name in os.listdir(full_directory):
                    # int(file_name[6:-4] : 현재의 z값
                    if (gt_min - limit_z) <= int(file_name[6:-4]) and (gt_max + limit_z) >= int(file_name[6:-4]): # 박테리아가 있는 z일 경우
                        self.data_list.append(os.path.join(full_directory, file_name)+"1") # 라벨을 1로 지정
                    if (gt_min - outlier_z) >= int(file_name[6:-4]) or (gt_max + outlier_z) <= int(file_name[6:-4]): # 박테리아가 없는 z일 경우
                        self.data_list.append(os.path.join(full_directory, file_name)+"0") # 라벨을 0로 지정

        self.validation = validation
        self.split = split
        self.train_rate = train_val_rate
        
        if self.split == 'train':
            train_len = int(round(len(self.data_list) * self.train_rate))
            if self.validation == False:
                self.data_list = self.data_list[:train_len]
                
            if self.validation == True:
                self.data_list = self.data_list[train_len:]
        
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        
    def __len__(self): 
        return len(self.data_list)
    
    def __getitem__(self, index): 
        image_path = self.data_list[index][:-1] 
        image = Image.open(image_path)
        label = int(self.data_list[index][-1])
        
        if (self.split =='train') and (self.validation==False):
            image = self.aug_transform(image)
            
        if (self.split =='test') or (self.validation==True):
            image = self.base_transform(image)
        
        return image, label
    
    base_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(), # (28, 28, 1) -> (1, 28, 28) // 0 ~ 255(grayscale) -> 0 ~ 1
                    transforms.Normalize(0.5, 0.5) # standard scaling # [0, 1] -> [-0.5, 0.5] -> [-1, 1]
                    ])

aug_transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.2), # 찌그러트리기
                    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.8, 1.2)), # 회전 이동 자동 적용
                    transforms.RandomAutocontrast(p=0.2), # 밝고 어두운 부분의 차이를 자동으로 조절
                    transforms.RandomHorizontalFlip(p=0.2), # 좌우반전
                    transforms.RandomVerticalFlip(p=0.2), # 상하반전
                    transforms.ToTensor(), # (28, 28, 1) -> (1, 28, 28) // 0 ~ 255(grayscale) -> 0 ~ 1
                    transforms.Normalize(0.5, 0.5) # standard scaling # [0, 1] -> [-0.5, 0.5] -> [-1, 1]
                    ])

def sampler(validation = False, train_val_rate = 0.9):
    df = pl.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv", columns=["tomo_id","Motor axis 0","Voxel spacing"], low_memory=True)
    labels = []  
    top_directory = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
    limit_angstroms = 1000 / 3 # 박테리아가 있다고 예측할 데이터 경계(angstroms)
    outlier_angstroms = 2000 
    for sub_directory in os.listdir(top_directory):
        full_directory = os.path.join(top_directory, sub_directory)
    
        df_current_tomo = df.filter(pl.col("tomo_id")==sub_directory)
        voxel_spacing = df_current_tomo.select("Voxel spacing")[0].item() 
        limit_z = m.trunc(limit_angstroms / voxel_spacing)
        outlier_z = m.trunc(outlier_angstroms / voxel_spacing) 
        gt_max = int(df_current_tomo.select("Motor axis 0").max().item())
        gt_min = int(df_current_tomo.select("Motor axis 0").min().item())
        
        if gt_max != -1:
            for file_name in os.listdir(full_directory):
                # int(file_name[6:-4] : 현재의 z값
                if (gt_min - limit_z) <= int(file_name[6:-4]) and (gt_max + limit_z) >= int(file_name[6:-4]): # 박테리아가 있는 z일 경우
                    labels.append(1) # 라벨을 1로 지정
                if (gt_min - outlier_z) >= int(file_name[6:-4]) or (gt_max + outlier_z) <= int(file_name[6:-4]): # 박테리아가 없는 z일 경우
                    labels.append(0) # 라벨을 0로 지정

    if validation == False:
        train_len = int(round(len(labels) * train_val_rate))
        labels = labels[:train_len]
        
    if validation == True:
        train_len = int(round(len(labels) * train_val_rate))
        labels = labels[train_len:]
        
    num_samples = len(labels) 

    class_counts = Counter(labels)

    # 클래스별 가중치 계산
    weights = [1.0 / class_counts[label] for label in labels] # 개수가 많은 라벨일 수록 적은 가중치가 적용됨
    sampler = WeightedRandomSampler(weights, 
                                    num_samples= num_samples, 
                                    replacement=True) # 중복 샘플링 허용 

    print(class_counts)

    return sampler

train_sampler = sampler(validation = False, train_val_rate = train_val_rate)
val_sampler = sampler(validation = True, train_val_rate = train_val_rate)

train_dataset = Classification_Dataset(root_dir='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025', split='train', validation=False, train_val_rate=train_val_rate, base_transform=base_transform, aug_transform=aug_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=train_sampler)

val_dataset = Classification_Dataset(root_dir='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025', split='train', validation=True, train_val_rate=train_val_rate, base_transform=base_transform, aug_transform=aug_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, sampler=val_sampler)
```


```python
# ResNet 모델 로드 
model = models.resnet34(weights=ResNet34_Weights.DEFAULT)

# 첫 번째 Conv 레이어 수정 (in_channels=1)
model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

# output을 binary로 변경
num_classes = 2
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

#  cpu 혹은 gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)


optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()
```


```python
class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=5, checkpoint_path='model.pt'):
        self.patience = patience            # 몇 번 참을지
        self.checkpoint_path = checkpoint_path

        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss, model):

        improved = False

        # 조건 1: 저장 조건 확인
        if (self.best_val_loss > val_loss) and (self.best_train_loss > train_loss):
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"Checkpoint saved! train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            self.best_val_loss = val_loss
            self.best_train_loss = train_loss
            self.counter = 0
            improved = True

        # 조건 2: 조기 종료 조건
        elif (val_loss > self.best_val_loss):
            self.counter += 1
            print(f"No improvement. train_loss={train_loss:.4f}, val_loss={val_loss:.4f}. Patience counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return improved

def train_model(model, criterion, optimizer, train_loader, epochs, val_dataloader, early_stopper, scheduler):
    
    start = time() #학습이 얼마나 걸리는 지 확인 (시작)

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_correct = 0
        total_val_loss = 0.0
        total_val_correct = 0
        for idx, data in enumerate(tqdm(train_loader)):
            model.train() 
            images, labels = data[0].to(device), data[1].to(device) 
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.max(outputs, dim=1)[1] 
            total_correct += (preds == labels).sum() 
            
            loss.backward()  
            optimizer.step() 
            scheduler.step()

        with torch.no_grad(): 
            model.eval() # evaluation mode # batch norm과 drop out의 모드 전환. 그 외에는 동일
            for data in tqdm(val_dataloader): 
                images, labels = data[0].to(device), data[1].to(device) 
                outputs = model(images) 
                
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.max(outputs, dim=1)[1] 
                total_val_correct +=  (preds == labels).sum() 
                
        train_avg_loss = total_loss / len(train_loader)
        train_avg_correct = total_correct / len(train_loader)

        val_avg_loss = total_val_loss / len(val_dataloader)
        val_avg_correct = total_val_correct / len(val_dataloader)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        wandb.log({"train loss": train_avg_loss, "train accuracy" : train_avg_correct, "validation loss": val_avg_loss, "validation accuracy" : val_avg_correct, "learning rate" : current_lr})
            
        improved = early_stopper(train_avg_loss, val_avg_loss, model)
            
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
        
                
    end = time() #학습이 얼마나 걸리는 지 확인 (끝)

    print("Training Done.")
    print(f"Elasped Time : {end-start:.4f} secs.")

```


```python
from IPython.display import FileLink
# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="simple-rule-project",
    # Set the wandb project where this run will be logged.
    project="BYU - Locating Bacterial Flagellar Motors 2025",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size" : batch_size,
        "architecture": "resnet34-binary-classification",
        "setting": "basic",
    },
)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.0001,                # 학습률 최고점
    steps_per_epoch=len(train_dataloader),  # 한 에폭 안의 스텝 수
    epochs=epochs,                 # 전체 epoch 수
    pct_start=0.01,             # 얼마나 빨리 최고점에 도달할지 (30% 지점에서 최고)
    anneal_strategy='cos',     # 감소 방식: 'cos' 또는 'linear'
    final_div_factor=1e4       # 마지막 lr은 max_lr / final_div_factor
)
early_stopper = EarlyStoppingAndCheckpoint(
    patience=5,
    checkpoint_path='best_classification_model.pt',
)

train_model(model, criterion, optimizer, train_dataloader, epochs, val_dataloader, early_stopper, scheduler)

wandb.save('best_classification_model.pt')
wandb.finish()

# 모델 로컬 다운
os.chdir(r'/kaggle/working')
FileLink(r'best_classification_model.pt')
```

### 오버레이 두 번째 문제
![image](/assets/images/BYU images/overlay2.png){: width="100%" height="100%"}

**binary classification** 모델을 완성하고 나서 tomogram마다 오버레이의 결과가 **천차만별**이라는 것을 알게 됐습니다. 오버레이가 잘 적용되는 이미지도 있지만 위 tomogram처럼 오버레이의 결과가 안좋은 경우도 있습니다. 이를 해결하기 위해서는 **오버레이의 수학식**을 **일일이** tomogram마다 **조정**해주어야 하는데 이는 **불가능(무의미)**합니다. 결국, **binary classification** 모델이 오버레이할 이미지를 잘 골라낸다고 해도 오버레이한 결과가 tomogram마다 상이하기에 **binary classification** 모델 **무용지물**이 되는 것 입니다. 이를 통해 항상 아이디어가 있다면 다른 상황에서 **예외**가 발생하지 않는지 **검증**을 미리해야 한다는 것을 깨달았습니다.

### U-Net 활용

이미지를 오버레이 하기 쉽도록 **U-Net**으로 **배경을 제거**하는 방법도 생각해봤습니다.

![image](/assets/images/BYU images/unet train data.png){: width="100%" height="100%"}

**규칙 기반(rule-based)**으로 하얀 바탕 이미지 위에 **타원, 사각형, 곡선을 무작위로 생성**한 이미지를 만듭니다. 그리고 그 위에 **가우시안 노이즈**를 뿌린 이미지를 따로 만듭니다. 데이터가 모두 만들어지면 **노이즈를 뿌린 이미지를 U-Net의 입력 데이터로, 노이즈를 뿌리지 않은 이미지를 정답 데이터로 학습**합니다.

![image](/assets/images/BYU images/unet output.png){: width="100%" height="100%"}

U-Net 모델의 학습 결과는 위와 같습니다. 어느정도 배경을 제거하기는 했지만 아무래도 수학적인 방식이 아니라 인공지능 모델이 복원했기 때문에 편모와 박테리아만 잘 남길 수 있다고 **신뢰하기는 어렵습니다**. 게다가 U-Net이 꽤 큰 모델이기 때문에, 한 tomogram당 300~600장 정도되는 이미지를 모두 inference하기에는 **cost 측면에서 어려움**이 많다고 느꼈습니다. 

결과적으로, 여러가지 한계로 인해 **오버레이 방식은 사용할 수 없다고 판단**하였습니다.

## 4. Yolo8 개조 - 다중 채널 입력

오버레이로 여러 이미지를 한 장으로 만드는 대신, yolo8 모델을 개조해서 여러 장의 이미지를 입력으로 받을 수 있도록 하는 방향으로 전향했습니다. 즉, 한 tomogram에 대한 **채널 사이즈가 1인 흑백 이미지들을 채널 방향으로 여러 장 쌓아서 하나의 데이터**를 이루도록 했습니다.\
그런데 한 tomogram에 대한 모든 이미지(slice)를 합치지 않고 **8장 간격**으로 **5장**의 이미지만 합쳤습니다 (e.g. 0, 8, 16, 24, 32). 그 이유는 뒤에 설명할 **Multi-head** 방식에서 **segmentation head**가 **1 채널**을 입력으로 받기 때문에 segmentation head의 입력과 너무 다른 입력 데이터가 들어오는 것을 방지하기 위함입니다.

yolo8 모델이 여러 장의 이미지를 입력으로 받을 수 있도록 ultralytics에서 제공하는 yolo8의 내부 코드를 직접 뜯어보고 개조하는 과정을 거쳤습니다. 그러나 yolo8이 여러 장의 이미지를 한 번에 받을 수 있게 되어도 여전히 (x, y) 좌표만 예측할 뿐 **z** 값까지는 예측하지 못했습니다. 그래서 **z를 예측할 수 있는 모델**을 따로 만들었습니다. 즉, x, y를 예측하는 yolo8 모델과 z를 예측하는 모델이 따로 존재합니다.\
yolo 모델이 출력한 x, y 좌표값을 중심으로 일정 범위만큼 이미지를 **잘라(crop)냅니다**. 그렇게 구한 더 작은 height, width를 가진 5채널 이미지를 z를 예측하는 모델에 주어 z값을 에측하게 됩니다. z를 예측하는 모델로, **attention layer**에 **ANN layer**를 결합한 모델과 **ResNet**에 **ANN layer**를 결합한 모델을 비교해 보았습니다. 그 결과 **ResNet**에 **ANN layer**를 결합한 모델의 성능이 더 뛰어났습니다.\
하지만 **yolo가 좌표를 잘못 예측**하게 되면 잘라낸 이미지에 모터의 이미지가 담겨있지 않음으로 절대 **z의 좌표를 제대로 찾을 수 없게 되는 방식**입니다.

## 5. Multi-head Yolo8
![image](/assets/images/BYU images/multi-head.png){: width="100%" height="100%"}

yolo8에는 모터의 좌표를 탐지하는 **Detection head**가 기본적으로 달려있습니다. 그런데 추가로 몸체, 편모, 모터를 모두 예측하는 **Segmentation head**를 달아서 둘을 동시에 학습시키면 모델의 성능이 더 좋아질 것이라고 가정을 했습니다. 

![image](/assets/images/BYU images/segment data.png){: width="100%" height="100%"}

그러나 주어진 데이터에는 segment를 위한 **라벨링**이 되어 있지 않기 때문에 제가 직접 규칙 기반(Rule-based)으로 **segmentation 데이터를 생성**하기로 했습니다. 위 이미지가 규칙 기반으로 만들어진 데이터 입니다. 각 이미지는 몸통, 편모, 모터에 대한 **segmentation mask**와 **bbox**가 함께 출력됩니다. 

**Detection head**는 입력으로 5채널을 받고 **Segmentation head**는 입력으로 1채널을 입력으로 받기 때문에 입력(CNN) 레이어도 Multi-head와 동일하게 **2개**가 존재합니다.

학습을 할 때만 **Multi-head**를 사용하고 학습이 완료되면 **Detection head**만 사용하게 됩니다.

아래가 **Multi-head yolo8** 모델을 학습시키는 코드입니다.

### 라이브러리


```python
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.optim as optim

import numpy as np
import polars as pl
import cv2
import os
import math
import time
import random
from tqdm.auto import tqdm
from scipy.special import comb

from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

'''
# detection label format
batch = {
    "batch_idx": tensor([0, 0, 1]),  # 3개의 GT가 있고, 2개는 이미지 0에, 1개는 이미지 1에 속함
    "cls": tensor([2, 5, 1]),        # 각각 클래스 2, 5, 1
    "bboxes": tensor([
        [0.1, 0.1, 0.2, 0.2], # xywh
        [0.3, 0.3, 0.5, 0.5],
        [0.4, 0.4, 0.6, 0.6]
    ])
}

# segmentation label format
batch = {
    "batch_idx": tensor([0, 0, 1, 2, 2]), # mask 제외 Detection과 동일
    "cls":       tensor([3, 7, 5, 1, 1]),
    "bboxes":    tensor([
                    [0.5, 0.5, 0.3, 0.4],
                    [0.2, 0.3, 0.2, 0.3],
                    [0.4, 0.4, 0.5, 0.5],
                    [0.6, 0.6, 0.1, 0.1],
                    [0.3, 0.2, 0.2, 0.3],
                 ]),
    "masks":     tensor([5, 160, 160])  # 각 마스크는 객체별로 2D 바이너리
}
'''
```

### Train  Dataset, Dataloader 구현


```python
class Detect3D_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir = "", validation=False, train_data_rate=1):
        super(Detect3D_Train_Dataset, self).__init__() # 상속

        self.data_list = []  # self.data_list에 모든 파일 목록 저장

        self.images_directory = os.path.join(root_dir, "images")
        self.labels_directory = os.path.join(root_dir, "labels")
        
        for sub_directory in os.listdir(self.images_directory):
            full_directory = os.path.join(self.images_directory, sub_directory)
            for file_name in os.listdir(full_directory):
                label_name = file_name[:-4] + ".txt"
                try:
                    with open(f"{self.labels_directory}/{sub_directory }/{label_name}", "r") as f:
                        labels = f.read()
                    self.data_list.append(os.path.join(full_directory, file_name) + "|" + labels)
                except FileNotFoundError:
                    self.data_list.append(os.path.join(full_directory, file_name) + "|")

        self.data_list = sorted(self.data_list)

        self.validation = validation
        self.train_rate = train_data_rate


        train_len = int(round(len(self.data_list) * self.train_rate))
        if self.validation == False:
            self.data_list = self.data_list[:train_len]
        else:
            self.data_list = self.data_list[train_len:]

    def transform(self, img: torch.Tensor, bboxes: torch.Tensor):
        """
        img: (5, H, W) - 5채널 이미지
        bboxes: (N, 4) - xywh 정규화 (0~1 범위)
        Returns: transformed img, transformed bboxes (정규화 유지)
        """
        #_, H, W = img.shape
        transform_type = random.choice(['none', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'])
    
        bboxes_clone = bboxes.clone()
    
        if transform_type == 'none':
            return img, bboxes
    
        if transform_type == 'hflip':
            img = torch.flip(img, dims=[2])  # W axis
            bboxes[:, 0] = 1.0 - bboxes_clone[:, 0]
        elif transform_type == 'vflip':
            img = torch.flip(img, dims=[1])  # H axis
            bboxes[:, 1] = 1.0 - bboxes_clone[:, 1]
        elif transform_type == 'rot90':
            img = torch.rot90(img, k=1, dims=[1, 2])
            x, y, w, h = bboxes_clone[:, 0], bboxes_clone[:, 1], bboxes_clone[:, 2], bboxes_clone[:, 3]
            bboxes[:, 0] = y
            bboxes[:, 1] = 1.0 - x
            bboxes[:, 2] = h
            bboxes[:, 3] = w
        elif transform_type == 'rot180':
            img = torch.rot90(img, k=2, dims=[1, 2])
            bboxes[:, 0] = 1.0 - bboxes_clone[:, 0]
            bboxes[:, 1] = 1.0 - bboxes_clone[:, 1]
        elif transform_type == 'rot270':
            img = torch.rot90(img, k=3, dims=[1, 2])
            x, y, w, h = bboxes_clone[:, 0], bboxes_clone[:, 1], bboxes_clone[:, 2], bboxes_clone[:, 3]
            bboxes[:, 0] = 1.0 - y
            bboxes[:, 1] = x
            bboxes[:, 2] = h
            bboxes[:, 3] = w
    
        return img, bboxes
        

    def __len__(self): # 데이터의 개수 # batch단위로 자르기 위함
        return len(self.data_list)

    def __getitem__(self, index):
        image_path_label = self.data_list[index]

        image_path, label = image_path_label.split("|")[0], image_path_label.split("|")[1]
        
        image = np.load(image_path)['arr_0']
        image = torch.tensor(image / 255).float()
        
        if label != "":
            label = label.strip().split()
            z = torch.tensor([int(label[i:i+5][-1][:-2]) for i in range(0, len(label), 5)], dtype=torch.uint8)
            bboxes = torch.tensor(np.array([label[i:i+5][:4] for i in range(0, len(label), 5)], dtype=np.float64))
            image, bboxes = self.transform(image, bboxes)
            label_dict = {"bboxes": bboxes, "z": z}

        elif label == "":
            label_dict = {"bboxes":[], "z":[]}

        return image, label_dict
    
def collate_fn(batch): # Yolov8의 loss function이 요구하는 형식에 맞게 dataloader의 출력을 변경
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    images = torch.stack(images, dim=0)

    batch_idx = [([i] * len(x["bboxes"])) for i, x in enumerate(labels)]
    batch_idx = torch.tensor(np.concatenate(batch_idx), dtype=torch.int16)
    cls = [([0] * len(x["bboxes"])) for x in labels]
    cls = torch.tensor(np.concatenate(cls), dtype=torch.int16)
    try:
        bboxes = torch.cat([x["bboxes"] for x in labels if len(x["bboxes"]) > 0], dim=0)
        z = torch.cat([x["z"] for x in labels if len(x["z"]) > 0], dim=0).type(torch.uint8)
    except: # 라벨이 없는 경우
        bboxes = torch.tensor([], dtype=torch.float32)
        z = torch.tensor([], dtype=torch.uint8)
    return images, {"batch_idx":batch_idx, "cls":cls, "bboxes":bboxes, "z":z}

def sampler(root_dir = "", validation = False, train_data_rate = 1): # class unbalance를 해소하기 위한 가중 샘플링
    from collections import Counter
    
    labels = []
    
    images_directory = os.path.join(root_dir, "images")
    labels_directory = os.path.join(root_dir, "labels")
    for sub_directory in os.listdir(images_directory):
        images_full_directory = os.path.join(images_directory, sub_directory)
        for file_name in os.listdir(images_full_directory):
            label_name = file_name[:-4] + ".txt"
            if os.path.isfile(f"{labels_directory}/{sub_directory }/{label_name}"):
                labels.append(f"{sub_directory}/{file_name}/1")
            else:
                labels.append(f"{sub_directory}/{file_name}/0")


    labels = sorted(labels)
    labels = [int(x[-1]) for x in labels]

    if validation == False:
        train_len = int(round(len(labels) * train_data_rate))
        labels = labels[:train_len]

    else:
        train_len = int(round(len(labels) * train_data_rate))
        labels = labels[train_len:]

    num_samples = len(labels)

    class_counts = Counter(labels)

    # 클래스별 가중치 계산
    weights = [1.0 / class_counts[label] for label in labels] # 개수가 많은 라벨일 수록 적은 가중치가 적용됨
    sampler = WeightedRandomSampler(weights,
                                    num_samples= num_samples,
                                    replacement=True) # 중복 샘플링 허용

    print(class_counts)

    return sampler

train_data_rate = 0.8
batch_size = 18 # 18
root_dir = "/kaggle/input/my-dataset"

def data_loader_process(root_dir, train_data_rate, batch_size):
    train_sampler= sampler(root_dir = root_dir, validation = False, train_data_rate = train_data_rate)
    val_sampler = sampler(root_dir = root_dir, validation = True, train_data_rate = train_data_rate)
    
    train_set = Detect3D_Train_Dataset(root_dir = root_dir, validation=False, train_data_rate=train_data_rate)
    val_set = Detect3D_Train_Dataset(root_dir = root_dir, validation=True, train_data_rate=train_data_rate)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)

    print("Train data 개수 :",len(train_set), "\n", "Validation data 개수 :", len(val_set),
          "\n", "Train batch 개수 :", len(train_loader), "\n", "Validation batch 개수 :", len(val_loader))
    return train_loader, val_loader
    
train_loader, val_loader = data_loader_process(root_dir, train_data_rate, batch_size)
```


```python
class Make_bacteria_data:
    def __init__(self):

        # must match height and width 
        self.img_H = 640    
        self.img_W = 640
        
        self.background_color = random.randint(190, 210) # 여러 개의 박테리아 이미지를 합치기 위해 박테리아의 색보다 배경 색이 무조건 더 커야함.
        self.body_color = random.randint(140, 160)
        self.border_color = random.randint(0, 20)
        self.flagella_color = random.randint(60, 80)

        self.base_body_size = 35

    def get_bbox_from_mask(self, mask: np.ndarray):
        """
        mask: np.ndarray of shape (H, W), binary mask with 0 (background) and 1 (object)
        return: (x, y, w, h) if object exists, else None
        """
        if mask.sum() == 0:
            return None  # No object in mask

        y_indices, x_indices = np.where(mask == 1)

        xmin = x_indices.min()
        xmax = x_indices.max()
        ymin = y_indices.min()
        ymax = y_indices.max()

        w = xmax - xmin + 1
        h = ymax - ymin + 1

        x_center = xmin + w / 2
        y_center = ymin + h / 2

        return [x_center, y_center, w, h]

    def make_body_points(self, center_point):
        # center_point, second_point == np.array([x, y])
        result = []
        random = np.random.randint(-1, 2, (1, 2))
        second_point = center_point + random
        result.append(center_point)
        result.append(second_point)
        for i in range(30):
            random = np.random.randint(-1, 2, (1, 2))
            result.append(result[-1] - (result[-2] - result[-1]) + random)

        return np.array(result)

    def bezier_curve(self, control_points, num_points=100):
        n = len(control_points) - 1
        ts = np.linspace(0, 1, num=num_points)
        curve = []

        for t in ts:
            point = np.zeros(2)
            for i in range(n + 1):
                bernstein = comb(n, i) * (1 - t)**(n - i) * t**i
                point += bernstein * control_points[i]
            curve.append(point)

        return np.array(curve, dtype= np.uint64)

    def perpendicular_points(self, p1, p2):
        """
        p1, p2: 두 점의 좌표 (튜플 또는 리스트) (x, y)
        distance: 수직선 상에서 교차 지점으로부터 떨어진 거리
        """
        # 벡터 방향 (p1 -> p2)
        direction = (p2 - p1)

        # 직각 방향 벡터 얻기 (90도 회전)
        perp_direction = np.array([-direction[1], direction[0]])

        # 정규화하여 단위 벡터로 만들기
        perp_unit = perp_direction / np.linalg.norm(perp_direction)

        distance = np.random.randint(-100, 100)

        # 거리만큼 떨어진 두 점 구하기
        point1 = p2 + perp_unit * distance
        point1 = np.clip(point1, 0, self.img_H)

        return point1.astype(np.uint64).tolist() #point2.tolist()

    def make_body_mask(self):

        num_bodys = np.random.choice([1, 2, 3], p=[0.8, 0.15, 0.05])
        if num_bodys == 3:
            tri_up_x, tri_up_y = int(self.img_W / 2), int(self.img_H / 3) - 30
            tri_left_x, tri_left_y = int(self.img_W / 3) - 30, int(self.img_H / 3*2) + 30 
            tri_right_x, tri_right_y = int(self.img_W / 3*2) + 30, int(self.img_H / 3*2) + 30
            center_points = np.array([[tri_up_x, tri_up_y], [tri_left_x, tri_left_y], [tri_right_x, tri_right_y]]).reshape(-1, 1, 2)
        elif num_bodys == 2:
            double_up_x, double_up_y = int(self.img_W / 3), int(self.img_H / 3)
            double_down_x, double_down_y = int(self.img_W / 3*2), int(self.img_H / 3*2)
            center_points = np.array([[double_up_x, double_up_y], [double_down_x, double_down_y]]).reshape(-1, 1, 2)
        elif num_bodys == 1:
            center_points = np.array([[int(self.img_H/2), int(self.img_W/2)]]).reshape(-1, 1, 2)

        body_masks = []
        body_bboxes = []
        for center_point in center_points:
            
            mask = np.zeros((self.img_H, self.img_W), dtype=np.uint8)

            body_points = self.make_body_points(center_point)

            body_weight = random.randint(0, 2)
            if body_weight == 0:
                r_weight = np.ones(len(body_points)).astype(np.uint8)
            if body_weight == 1:
                r_weight = (np.arange(0, len(body_points)) / 12 + 1).astype(np.uint8)
            if body_weight == 2:
                r_weight = (np.arange(0, len(body_points))[::-1] / 12 + 1).astype(np.uint8)

            for j, x in enumerate(body_points):
                cv2.circle(mask, x[0], self.base_body_size * r_weight[j], 1, -1)

            body_masks.append(mask)

            body_bboxes.append(self.get_bbox_from_mask(mask))

        return body_masks, body_bboxes, body_weight

    def mask_body_border(self, mask, body_weight):

        img_H, img_W = mask.shape

        # 경계선: erosion 후 원래 마스크와 차이 계산
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        inner_border = mask - eroded  # 테두리만 남음

        if body_weight > 0:
            boader_dist = random.randint(5, 10)
        else:
            boader_dist = 5
        dilated1 = cv2.dilate(mask, kernel, iterations=boader_dist)
        dilated2 = cv2.dilate(mask, kernel, iterations=boader_dist+1)
        outer_border = dilated2 - dilated1 - mask    # 테두리만 남음

        # 시각화용 결과 이미지 (흰색 배경)
        
        image = np.ones((img_H, img_W), dtype=np.uint8) * self.background_color
   
        image[mask == 1] = self.body_color        # 도형 내부를 회색으로
        image[inner_border == 1] = self.border_color         # 경계선은 검정으로
        image[outer_border == 1] = self.border_color         # 경계선은 검정으로
        
        return image, inner_border, outer_border, boader_dist

    def make_flagella_and_mask(self, body_image, inner_border, outer_border, boader_dist):
        inner_border_point = np.argwhere(inner_border == 1)
        inner_border_point_randidx = np.random.randint(inner_border_point.shape[0])
        inner_border_randpoint = inner_border_point[inner_border_point_randidx]

        outer_border_point = np.argwhere(outer_border == 1)
        distances = np.linalg.norm(outer_border_point - inner_border_randpoint, axis=1) # 거리 계산 (유클리드 거리)
        closest_idx = np.argmin(distances) # 가장 가까운 인덱스
        closest_point = outer_border_point[closest_idx]

        dirction = (inner_border_randpoint - closest_point)    
        dirction_unit = dirction / np.linalg.norm(dirction)
        point3 = closest_point - 10 * dirction_unit
        point4 = self.perpendicular_points(point3, closest_point - 30 * dirction_unit)
        point5 = self.perpendicular_points(point3, closest_point - 50 * dirction_unit)
        point6 = self.perpendicular_points(point3, closest_point - 70 * dirction_unit)

        control_points = np.array([list(inner_border_randpoint), list(closest_point), list(point3), list(point4), list(point5), list(point6)])
        curve_points = self.bezier_curve(control_points, num_points=100)
        curve_points = curve_points.reshape(-1,1,2)[:,:,::-1]
        
        valid_mask = np.all((curve_points >= 0) & (curve_points <= self.img_H), axis=(1, 2))
        curve_points = curve_points[valid_mask] # 조건을 만족하는 원소만 추출

        
        flagella_thickness = 1 * boader_dist - 2
        cv2.polylines(body_image, [curve_points], isClosed=False, color=self.flagella_color, thickness=flagella_thickness)

        flagella_mask = np.array(cv2.polylines(np.zeros((self.img_H, self.img_W), dtype=np.uint8), [curve_points.astype(np.int32)], isClosed=False, color=1, thickness=flagella_thickness))
        flagella_bbox = self.get_bbox_from_mask(flagella_mask)

        motor_mask = np.array(cv2.circle(np.zeros((self.img_H, self.img_W)), inner_border_randpoint[::-1], 5, 1, -1), dtype=np.uint8)
        motor_bbox = self.get_bbox_from_mask(motor_mask)

        return body_image, flagella_mask, flagella_bbox, motor_mask, motor_bbox

    def make_image(self):
        body_masks, body_bboxes, body_weight = self.make_body_mask()
        flagella_masks = []
        flagella_bboxes = []
        motor_masks = []
        motor_bboxes = []
        images = []

        for i in body_masks:
            image,  inner_border, outer_border, boader_dist = self.mask_body_border(i, body_weight)
            
            num_flagella = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
            for j in range(num_flagella):
                image, flagella_mask, flagella_bbox, motor_mask, motor_bbox= self.make_flagella_and_mask(image, inner_border, outer_border, boader_dist) 
                flagella_masks.append(flagella_mask)
                flagella_bboxes.append(flagella_bbox)
                motor_masks.append(motor_mask)
                motor_bboxes.append(motor_bbox)
                
            images.append(image)

        if len(body_masks) == 1:
            image = images[0]
        if len(body_masks) > 1:
            image = np.where(images[0] < self.background_color, images[0], images[1])
        if len(body_masks) > 2:
            image = np.where(image < self.background_color, image, images[2])

        cls = [] # body: 0, flagella: 1, motor: 2
        masks = []
        bboxes = []

        for i, j in zip(body_masks, body_bboxes):
            masks.append(i)
            bboxes.append(j)
            cls.append(0)

        for i, j in zip(flagella_masks, flagella_bboxes):
            masks.append(i)
            bboxes.append(j)
            cls.append(1)

        for i, j in zip(motor_masks, motor_bboxes):
            masks.append(i)
            bboxes.append(j)
            cls.append(2)

        cls = torch.tensor(cls)
        masks = torch.tensor(np.array(masks))
        bboxes = np.array(bboxes)
        bboxes[:,[0, 2]] = bboxes[:,[0, 2]] / self.img_W # width
        bboxes[:,[1, 3]] = bboxes[:,[1, 3]] / self.img_H # height
        bboxes = torch.tensor(bboxes)

        return image, cls, masks, bboxes

    def add_noise(self, image):
        # 가우시안 노이즈 추가
        mean = 0
        std = random.randint(70, 80)  # 표준편차, 노이즈 세기
        noise = np.random.normal(mean, std, image.shape)

        noisy_img = image + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return noisy_img

    def add_noise_ellipse(self, image):
        # 타원 그리기
        num_ellipse = random.randint(0, 30)
        color = random.randint(20, 70)

        for _ in range(num_ellipse):
            center = np.random.randint(0, self.img_H, (2))
            axes = np.random.randint(1, 8, (2))          # 반가로: 100, 반세로: 50
            angle = random.randint(0, 90)                # 회전 각도

            cv2.ellipse(image, center, axes, angle, 0, 360, color=50, thickness=-1)

        return image
    
    def __call__(self):
        image, cls, masks, bboxes = self.make_image()
        image = self.add_noise(image)
        image = self.add_noise_ellipse(image)
        image = torch.tensor(image)

        return image, cls, masks, bboxes

    def batch_generate(self, batch_size=8):
        image_lst = []
        cls_lst = []
        masks_lst = []
        bboxes_lst = []
        batch_idx = []
        
        for i in range(batch_size):
            image, cls, masks, bboxes = make_bacteria()
            image_lst.append(image.unsqueeze(0) / 255)
            cls_lst.append(cls)
            masks_lst.append(masks)
            bboxes_lst.append(bboxes)
            batch_idx.append(torch.zeros(len(cls)) + i)
        
        batch_idx = torch.cat(batch_idx)
        image_lst = torch.stack(image_lst, dim=0)
        cls_lst = torch.cat(cls_lst)
        bboxes_lst = torch.cat(bboxes_lst, dim=0)
        masks_lst = torch.cat(masks_lst, dim=0)
        
        label = {
            "batch_idx" : batch_idx,
            "cls" : cls_lst,
            "bboxes" : bboxes_lst,
            "masks": masks_lst
             }
        
        return image_lst, label
    
    def image_show(self, figsize=(6,6)):
        import matplotlib.pyplot as plt
        image, _, _, _ = self.__call__()

        plt.figure(figsize=figsize)
        plt.imshow(image, cmap='gray')

        plt.axis('off')
        plt.show()
        
```


```python
a = Make_bacteria_data()
a.image_show()
```

![image](/assets/images/BYU images/segment data.png){: width="100%" height="100%"}


```python
def make_z_label(data):
    if data[1]["batch_idx"].numel() != 0:
        crop_size = 80
        crop_half_size = round(crop_size / 2)
        cropped_images = []
        vit_labels = []
        batch_idx = data[1]['batch_idx']
        for i in range(len(batch_idx)):
            z = (data[1]["z"][i] / 3).int()
            vit_labels.append(z)

            bbox = data[1]["bboxes"][i].clone()
            image = data[0][batch_idx[i].item()].clone()
            img_height, img_width = 640, 640
            x_center, y_center, _, _ = bbox
            x_center *= img_width
            y_center *= img_height
            x_center, y_center = int(x_center), int(y_center)
            
            y_min = y_center-crop_half_size
            y_max = y_center+crop_half_size
            x_min = x_center-crop_half_size
            x_max = x_center+crop_half_size
            
            if (y_min<0) | (y_max>640) | (x_min<0) | (x_max>640):
                
                back_ground = torch.zeros(5, crop_size, crop_size) + 0.5411
                y_min_attach_idx, y_max_attach_idx, x_min_attach_idx, x_max_attach_idx = 0, crop_size, 0, crop_size
                
                if y_min < 0:
                    y_min_attach_idx = abs(y_min)
                    y_min = 0
                if y_max > 640:
                    y_max_attach_idx = crop_size - (y_max - 640)
                    y_max = 640
                if x_min < 0:
                    x_min_attach_idx = abs(x_min)
                    x_min = 0
                if x_max > 640:
                    x_max_attach_idx = crop_size - (x_max - 640)
                    x_max = 640
                    
                image = image[:, y_min:y_max, x_min:x_max]
                back_ground[:, y_min_attach_idx:y_max_attach_idx, x_min_attach_idx:x_max_attach_idx] = image
                image = back_ground
                
            else:
                image = image[:, y_min:y_max , x_min:x_max]
                
            cropped_images.append(image)
        cropped_images = torch.stack(cropped_images, dim=0).type(torch.float)
        vit_labels = torch.stack(vit_labels, dim=0).type(torch.int64)
    else:
        cropped_images = None
        vit_labels = None
        
    return cropped_images, vit_labels
```


```python
cropped_images, vit_labels = make_z_label(data)

batch_index = 2 
imgs = []
for i in range(5):
    imgs.append(cropped_images[batch_index][i])
print("정답 라벨에 가장 가까운 이미지 :", int((vit_labels[batch_index] / 8 + 0.1).round() +1), "번째")
print("진짜 정답 라벨 :", int(vit_labels[batch_index] + 1), "/33 번째")

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12,12))

#plt.axis('off')
for i in range(5):
    axes[i].imshow(imgs[i], cmap='gray')
```

![image](/assets/images/BYU images/crop.png){: width="100%" height="100%"}

### 모델 구현

**Yolov8 정의**


```python
# 모델
# -----------------------------
# Utility Modules
# -----------------------------
def make_anchors(feats, strides, grid_cell_offset=0.5):  # x, self.stride, 0.5
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides): # [8.0, 16.0, 32.0]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1])) # size별로 feature의 h,w를 가져옴
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x  # 0.5 ~ 79.5
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y  # 0.5 ~ 79.5
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") #if TORCH_1_10 else torch.meshgrid(sy, sx) # (w x h) size의 0.5 ~ 79.5로 채워진 행렬 2개
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2)) # 마지막에 차원을 하나 추가해서 해당 차원으로 sx,sy를 쌓음-> (80,80,2). 마지막 차원을 유지하고 2차원으로 만듬->(6400,2)
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device)) # stride로 채워진 ((h*w), 1)크기의 텐서 생성 # 8:6400개, 16:1600개, 32:400개 -> (8400, 1)
    return torch.cat(anchor_points), torch.cat(stride_tensor) 
        # torch.cat(anchor_points): (6400,2), (1600,2), (400,2)를 0방향으로 concat하여 (8400,2)
        # torch.cat(stride_tensor): (6400,1), (1600,1), (400,1)를 0방향으로 concat하여 (8400,1)

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float) # [0.0, 1.0, ..., 15.0] 정수를 (학습하지 않는)파라미터로
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) 
        self.c1 = c1

    def forward(self, x):
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a) #softmax를 확률값으로 하여 [0.0, 1.0, ..., 15.0]의 기대값을 구함


# -----------------------------
# Model Modules
# -----------------------------
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
    
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        c1, c2 = int(c1), int(c2)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2) ##### [c, h, w]!!!!!!!
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=3, e=0.5):
        """
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k, 1)
        self.cv2 = Conv(c_, c2, k, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c1, c2, n = int(c1), int(c2), int(n)
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1)) # B,C,H,W -> C-wise로 2등분해서 튜플로 만듬
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        
class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):
        """
        Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


# -----------------------------
# Input layer
# -----------------------------
class Input_layer(nn.Module):
    def __init__(self, w, det_3d_dim=5):
        super().__init__()
        self.detect3d = Conv(det_3d_dim, 64*w, 3, 2)
        self.segment = Conv(1, 64*w, 3, 2)
        self.det_3d_dim = det_3d_dim
    
    def forward(self, x):
        if YOLOv8.mode == "detect":
            if x.size()[1] != self.det_3d_dim:
                raise Exception(f"Expected input channel size: {self.det_3d_dim}. but you got {x.size()[1]}")
            x = self.detect3d(x)
        if YOLOv8.mode == "segment":
            if x.size()[1] != 1:
                raise Exception(f"Expected input channel size: {1}. but you got {x.size()[1]}")
            x = self.segment(x)
        if YOLOv8.mode not in ["detect","segment"]:
             raise Exception("Enter the mode properly. 'detect' or 'segment'")
        return x

# -----------------------------
# Backbone
# -----------------------------
class Backbone(nn.Module):
    def __init__(self, model_scale):
        super().__init__()
        d, w, r = model_scale[0], model_scale[1], model_scale[2]
        self.input_layer = Input_layer(w)
        self.stage1 = nn.Sequential(Conv(64*w, 128*w, 3, 2), C2f(128*w, 128*w, 3*d, shortcut=True)) 
        self.stage2 = nn.Sequential(Conv(128*w, 256*w, 3, 2), C2f(256*w, 256*w, 6*d, shortcut=True))
        self.stage3 = nn.Sequential(Conv(256*w, 512*w, 3, 2), C2f(512*w, 512*w, 6*d, shortcut=True)) 
        self.stage4 = nn.Sequential(Conv(512*w, 512*w*r, 3, 2), C2f(512*w*r, 512*w*r, 3, shortcut=True), SPPF(512*w*r, 512*w*r)) 

    def forward(self, x):
        x = self.input_layer(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x2, x3, x4]  # C3, C4, C5 feature maps
        
# -----------------------------
# Neck
# -----------------------------
class Neck(nn.Module):
    def __init__(self, model_scale):
        super().__init__()
    
        d, w, r = model_scale[0], model_scale[1], model_scale[2]
    
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f1 = C2f(512*w*(1+r), 512*w, 3*d, shortcut=False)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f2 = C2f(768*w, 256*w, 3*d, shortcut=False)

        self.down_conv1 = Conv(256*w, 256*w, 3, 2)
        self.c2f3 = C2f(768*w, 512*w, 3*d, shortcut=False)

        self.down_conv2 = Conv(512*w, 512*w, 3, 2)
        self.c2f4 = C2f(512*w*(1+r), 512*w*r, 3*d, shortcut=False)

    def forward(self, x):
        return x
        c3, c4, c5 = x
        p5 = c5
        p4 = self.c2f1(torch.cat([self.upsample1(p5), c4], dim=1))
        p3 = self.c2f2(torch.cat([self.upsample2(p4), c3], dim=1))

        n4 = self.c2f3(torch.cat([self.down_conv1(p3), p4], dim=1))
        n5 = self.c2f4(torch.cat([self.down_conv2(n4), p5], dim=1))

        return [p3, n4, n5]

# -----------------------------
# Detect Head
# -----------------------------
class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    def __init__(self, nc=1, ch=()):
        """Initialize the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([8, 16, 32]) 
        c2, c3 =   max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = (nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch))
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) # input 채널별로 좌표, 분류 output concat
        if YOLOv8.inference == False:
            return x # 학습이면 그대로 출력
        return self._inference(x)

    def _inference(self, x):
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (List[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) # H, nc + reg_max * 4, -1

        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        self.shape = shape
            
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) # 박스 좌표를 나타내는 feature와 분류하는 feature 분리

        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides # * self.strides: DFL로 정규화된 bbox 좌표를 원래 이미지 좌표로 변환

        return torch.cat((dbox, cls.sigmoid()), 1) #classificatoin을 0~1범위로 # (B, 4+c, a)
    
    def decode_bboxes(self, bboxes, anchor_points, xywh=True, dim=1): # bboxes: (batch, 4, anchors),  anchor_points: (1, 2, anchors) # dim=1 
        """Transform bboxes(ltrb) to box(xywh or xyxy)."""
        lt, rb = bboxes.chunk(2, dim) 
        x1y1 = anchor_points - lt # bbox의 좌측 상단(?) 좌표  # [x - l, y - t]
        x2y2 = anchor_points + rb # bbox의 우측 하단(?) 좌표  # [x + r, y + b]
        if xywh:
            c_xy = (x1y1 + x2y2) / 2 # bbox의 중심
            wh = x2y2 - x1y1 # width, height로 변환
            return torch.cat((c_xy, wh), dim)  # xywh bbox (batch, 4, anchors)
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox (batch, 4, anchors)
        
    def bias_init(self): # 매우 중요한 초기값 설정. 이를 설정할 때와 하지 않았을 때의 성능 차이가 매우 큼.
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
                # box 회귀 부분의 bias를 모두 1로 초기화. 보통 중심 (cx, cy), 크기 (w, h)에 대한 초기값을 조금 조정함으로써 처음부터 박스가 전혀 예측되지 않는 현상을 피하기 위함.
                # 예측 box의 위치가 학습 초기에 완전히 무의미하지 않도록 하고, anchor-free 구조에서 좀 더 빠른 수렴을 유도.
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
                # 확률(cls)에 대한 bias 초기값. math.log(p / (1 - p)) 형태의 로짓 초기화로, 아주 희귀한 물체가 있다는 가정. 실제로는 전체 이미지 중 약 0.01 (=1%)만이 물체를 포함한다고 가정.
                # 학습 초기에 과도한 false positive를 줄이고, objectness + class confidence가 너무 높게 나오는 것을 막아 안정된 수렴을 유도함.
       
    
# -----------------------------
# Segment Head
# -----------------------------
class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32): # ch[0], self.npr, self.nm
        """
        Initialize the YOLOv8 mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=3, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos   # x[0].shape == (b, 256*w, 80, 80)
        bs = p.shape[0]  # batch size  # P.shape == (b, nm, 160, 160)

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        
        x = Detect.forward(self, x)
        if YOLOv8.inference == False:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) 


# -----------------------------
# YOLOv8 Full Model
# -----------------------------
class YOLOv8(nn.Module):
    mode = "detect"
    inference = False
    def __init__(self, scale="l", det_num_classes=1, seg_num_classes=3):
        super().__init__()
        self.det_nc = det_num_classes
        self.seg_nc = seg_num_classes
        self.args = {
            'box': 0.05,   # localization loss (Bbox)
            'cls': 0.5,    # classification loss
            'dfl': 1.5     # distribution focal loss (bounding box refinement)
        }
        scale = scale 
        # [depth_multiple, width_multiple, ratio]
        model_scales = {"n":[0.33, 0.25, 2.0], 
                        "s":[0.33, 0.50, 2.0],
                        "m":[0.67, 0.75, 1.5],
                        "l":[1.00, 1.00, 1.0],
                        "x":[1.00, 1.25, 1.0]}
        model_scale = model_scales[scale] 
        _, w, r = model_scale[0], model_scale[1], model_scale[2]
        
        self.backbone = Backbone(model_scale = model_scale)
        self.neck = Neck(model_scale = model_scale)
        
        self.detect = Detect(nc= det_num_classes, ch=(int(256*w), int(512*w), int(512*w*r)))
        self.segment = Segment(nc= seg_num_classes, ch=(int(256*w), int(512*w), int(512*w*r)))

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        if self.mode == "detect" :
            return self.detect(x)
        elif self.mode == "segment":
            return self.segment(x)
        else:
            raise ValueError("Unsupported mode. choose in 'detect', 'segment'")
        # return fpn_feats
```


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
yolo = YOLOv8(scale="l", det_num_classes=1, seg_num_classes=3).to(device)
yolo.detect.bias_init()
yolo.segment.bias_init()
YOLOv8.mode = "detect"

state_dict = torch.load("/kaggle/input/multi-head-trained-yolov8/pytorch/default/2/4epoch_best_multi_head_trained_yolo.pt", weights_only=True)
yolo.load_state_dict(state_dict)
```

**Z를 예측하는 ResNetMLP 정의**


```python
class ResNetMLP(nn.Module):
    def __init__(self, 
                 num_classes=11, 
                 in_chans=1, 
                 embed_dim = 512,
                 num_patch=5,
                 dropout=0.2):
        super().__init__()

        # 1. ResNet18 feature extractor
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.patch_encoder = create_feature_extractor( # backbone의 중간 출력을 추출하는 기능
            backbone,
            return_nodes={"avgpool": "feat"} # dict -> key :"feat", value : avgpool layer output
        )
        self.embed_dim = embed_dim

        # 6. Classification head
        self.head = nn.Sequential(
            nn.Linear(num_patch * self.embed_dim, num_patch * self.embed_dim),
            nn.BatchNorm1d(num_patch * self.embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_patch * self.embed_dim, num_patch * self.embed_dim),
            nn.BatchNorm1d(num_patch * self.embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_patch * self.embed_dim, num_patch * self.embed_dim),
            nn.BatchNorm1d(num_patch * self.embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(num_patch * self.embed_dim, num_classes)  # hidden_layer -> output_layer
        )


    def forward(self, x):
        """
        x: (B, N, H, W) — B=batch, N=patches, H,W=patch size
        """
        B, N, H, W = x.shape

        # Flatten into B*N fake-batch
        x = x.view(-1, 1, H, W)  

        feats = self.patch_encoder(x)["feat"]  # (B*N, 512, 1, 1)
        feats = feats.view(B, N*self.embed_dim)  # (B, N*D)
        
        logits = self.head(feats)  # (B, num_classes)

        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

resnet_mlp = ResNetMLP(
    num_classes=11,
    in_chans=1).to(device)

```

### Loss function 구현

**Detect loss function**


```python
# Loss function

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        bg_idx (int): Background class index.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9): #(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)

        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
            # pred_scores.detach().sigmoid(), # (batch, anchors, c)
            # (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), # (b, h*w, 4)
            # anchor_points * stride_tensor, # (2, anchors)
            # gt_labels, # (B, counts.max, 1)
            # gt_bboxes, # (B, counts.max, 4)
            # mask_gt, # (B, counts.max, 1)
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0] # batch_size
        self.n_max_boxes = gt_bboxes.shape[1] 
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # (B, num_max_boxes, anchors)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt): # (pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2] # num_total_anchors
        mask_gt = mask_gt.bool()  # mask_gt = mask_in_gts * mask_gt -> (b, max_max_boxes, a) # 1과 0으로 이뤄진 mask를 boolean으로 변환
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device) # (b, nmb, a)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device) # (b, nmb, a)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # (2, b, nmb)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # (b, nmb)
        ind[1] = gt_labels.squeeze(-1)  # (b, nmb)
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size,
                              max_num_obj is the maximum number of objects, and h*w represents the
                              total number of anchor points.
            largest (bool): If True, select the largest values; otherwise, select the smallest values.
            topk_mask (torch.Tensor): An optional boolean tensor of shape (b, max_num_obj, topk), where
                                topk is the number of top candidates to consider. If not provided,
                                the top-k values are automatically computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            target_labels (torch.Tensor): Shape (b, h*w), containing the target labels for positive anchor points.
            target_bboxes (torch.Tensor): Shape (b, h*w, 4), containing the target bounding boxes for positive
                                          anchor points.
            target_scores (torch.Tensor): Shape (b, h*w, num_classes), containing the target scores for positive
                                          anchor points.
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9): # (anc_points, gt_bboxes)
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability. Defaults to 1e-9.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0] # num_anchors
        bs, n_boxes, _ = gt_bboxes.shape # n_boxes = num_max_boxes
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1) # (bs, n_boxes, a, 4)
            # x[None]: 맨 앞에 차원 추가 
            # xy_centers[None] -> (1, a, 2) // lt or rb -> (bs*n_boxes, 1, 2)  # xy_centers[None] - lt -> (bs*n_boxes, a, 2) -# cat -> (bs*n_boxes, a, 4)
        return bbox_deltas.amin(3).gt_(eps) # (bs, n_boxes, a)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos

class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y

def dist2bbox(distance, anchor_points, xywh=True, dim=-1): # distance: (batch, 4, anchors),  anchor_points: (1, 2, anchors) # dim=1 
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim) 
    x1y1 = anchor_points - lt # bbox의 좌측 상단(?) 좌표  # [x - l, y - t]
    x2y2 = anchor_points + rb # bbox의 우측 하단(?) 좌표  # [x + r, y + b]
    if xywh:
        c_xy = (x1y1 + x2y2) / 2 # bbox의 중심
        wh = x2y2 - x1y1 # width, height로 변환
        return torch.cat((c_xy, wh), dim)  # xywh bbox (batch, 4, anchors)
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox (batch, 4, anchors)

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, args, model_head, tal_topk=10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model_head.parameters()).device  # get model device
        h = args  # hyperparameters
        m = model_head  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device) # [0,1,...,16]

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape # nl=num_GT, ne=1+1+4
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device) # (b, 0, 5) -> 차원에 0이 있으면 빈 텐서 반환
        else:
            i = targets[:, 0]  # image index 
            _, counts = i.unique(return_counts=True) # 유일값과 그 유일값이 원래 몇개있었는지를 함께 반환 
                # torch.tensor([1,1,2,2,3,4]).unique(return_counts=True) -> (tensor([1, 2, 3, 4]), tensor([2, 2, 1, 1]))
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device) # (B, counts.max, 5) # counts.max: 한 이미지가 가진 GT 개수 중 가장 많은 GT 개
            for j in range(batch_size): # image idx는 batch 기준으로 0~batch_zie로 지정하는 것으로 보임
                matches = i == j  # 텐서 i의 원소와 스칼라 j를 비교하여 boolean 텐서 반환 # j번째 이미지가 가지고 있는 GT들을 볼 수 있는 mask
                if n := matches.sum(): # := -> 조건문에서 변수에 할당할 수 있는 월러스 연산자. # GT가 없으면 0임으로 False 하나 이상 있으면 True
                    out[j, :n] = targets[matches, 1:] # n: j번째 이미지가 가지고 있는 GT 수 # traget에서 j번째 이미지에 해당하는 cls와 boxes를 가져온다.
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor)) # out[..., 1:5]->(B, counts.max, 4) /// scale_tensor->(4)
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype)) # (b, a, 4, 16) -> (b, a, 4, 0) -> (b, a, 4)
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False) # (b, a, 4)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds # 리스트임으로 preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(   # feats[0].shape[0]==batch
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # (batch, c, anchors) -> (batch, anchors, c)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() # (batch, 16*4, anchors) -> (batch, anchors, 16*4)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5) # (2, anchors(8400))

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) # -> (num_GT, 1+1+4)
            # batch = {"batch_idx": tensor([0, 0, 1]),  # 3개의 GT가 있고, 2개는 이미지 0에, 1개는 이미지 1에 속함
            #          "cls": tensor([2, 5, 1]),        # 각각 클래스 2, 5, 1
            #          "bboxes": tensor([
            #             [0.1, 0.1, 0.2, 0.2],
            #             [0.3, 0.3, 0.5, 0.5],
            #             [0.4, 0.4, 0.6, 0.6]])}
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) # imgsz[[1, 0, 1, 0]] = (w,h,w,h) # (B, counts.max, 1+4)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0) # (B, counts.max, 4) -> (B, counts.max, 1) # x.gt_(y) : x > y면 1 아니면 0 # GT가 있으면 1아니면 0

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(), # (batch, anchors, c)
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype), # (b, h*w, 4)
            anchor_points * stride_tensor, # (2, anchors)
            gt_labels, # (B, counts.max, 1)
            gt_bboxes, # (B, counts.max, 4)
            mask_gt, # (B, counts.max, 1)
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["cls"]  # cls gain
        loss[2] *= self.hyp["dfl"]  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
        
det_criterion = v8DetectionLoss(yolo.args, yolo.detect)
```

**Segment loss function**


```python
def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def crop_mask(masks, boxes):
    """
    Crop masks to bounding boxes.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks.
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form.

    Returns:
        (torch.Tensor): Cropped masks.
    """
    _, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, args, model_head):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(args, model_head)
        self.overlap = False

    def __call__(self, preds, batch):
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(   
            (self.reg_max * 4, self.nc), 1
        )
 
        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous() # (batch, c, anchors) -> (batch, anchors, c)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous() # (batch, 16*4, anchors) -> (batch, anchors, 16*4)
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0] # target mask를 proto의 h,w에 맞게 보간

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp["box"]  # box gain
        loss[1] *= self.hyp["box"]  # seg gain
        loss[2] *= self.hyp["cls"]  # cls gain
        loss[3] *= self.hyp["dfl"]  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()
        
seg_criterion = v8SegmentationLoss(yolo.args, yolo.segment)
```

**ResNetMLP loss function**


```python
resmlp_criterion = torch.nn.CrossEntropyLoss()
```

### Training


```python
class EarlyStoppingAndCheckpoint:
    def __init__(self, patience=10, model_name="", improvement_path='model.pt', no_improvement_path='temp_model.pt'):
        self.model_name = model_name
        self.patience = patience            # 몇 번 참을지
        self.improvement_path = improvement_path
        self.no_improvement_path = no_improvement_path

        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, val_loss, model, epoch, loss_text):
        if self.early_stop == False:
            
            # 조건 1: 성능 향상 모델 저장 조건 확인
            val_loss_sum = sum(val_loss)
            train_loss_sum = sum(train_loss)
            if (self.best_val_loss > val_loss_sum) and (self.best_train_loss > train_loss_sum):
                torch.save(model.state_dict(), self.improvement_path)
                wandb.save(self.improvement_path)
                
                #print(f"Checkpoint saved! train_loss={train_loss:.5f}, val_loss={val_loss:.5f}")
                print(f"{self.model_name}. Checkpoint saved! Epoch {epoch+1}.","\n", 
                      "train_loss => ",[f"{t}:{x:.5f}" for t, x in zip(loss_text, train_loss)],"\n",
                      "val_loss => ",[f"{t}:{x:.5f}" for t, x in zip(loss_text, val_loss)]
                     )
                self.best_val_loss = val_loss_sum
                self.best_train_loss = train_loss_sum
                self.counter = 0
    
            # 조건 2: 조기 종료 조건
            elif (val_loss_sum > self.best_val_loss):
                
                torch.save(model.state_dict(), self.no_improvement_path)
                wandb.save(self.no_improvement_path)
                self.counter += 1
                
                #print(f"No improvement. train_loss={train_loss:.5f}, val_loss={val_loss:.5f}. Patience counter: {self.counter}/{self.patience}")
                print(f"{self.model_name}. No improvement. Epoch {epoch+1}.","\n", 
                  "train_loss => ",[f"{t}:{x:.5f}" for t, x in zip(loss_text, train_loss)],"\n",
                  "val_loss => ",[f"{t}:{x:.5f}" for t, x in zip(loss_text, val_loss)],
                    f"Patience counter: {self.counter}/{self.patience}"
                 )
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            print(f"{self.model_name}. early stopped.")
```


```python
# detect & segment 
def train_model(yolo, epochs, 
                det_criterion, seg_criterion, 
                optimizer,
                train_loader, val_loader, 
                early_stopper,
                scheduler,
                make_bacteria
               ):
    
    start = time.time() #학습이 얼마나 걸리는 지 확인 (시작)
    
    det_loss_text = ["box loss", "class loss", "dfl loss"]
    #seg_loss_text = ["box loss", "seg loss", "class loss", "dfl loss"]
    
    for epoch in tqdm(range(epochs)):
        yolo.train() 
        det_train_total_loss = torch.tensor([0., 0., 0.])
        seg_train_total_loss = torch.tensor([0., 0., 0., 0.])

        for data in  tqdm(train_loader): 
            images, labels = data[0].to(device), data[1]
     
            #---------------------------
            # Detect training
            #---------------------------
            YOLOv8.mode = "detect"
            outputs = yolo(images)
            
            det_loss = det_criterion(outputs, labels) 
            det_loss_rec = det_loss[1].to('cpu') 
            
            det_train_total_loss += det_loss_rec

            det_loss[0].sum().backward() 
            optimizer.step() 
            optimizer.zero_grad()

            #---------------------------
            # Segment training
            #---------------------------
            YOLOv8.mode = "segment"
            images, labels = make_bacteria.batch_generate(batch_size = 4)
            images = images.to(device)
            
            outputs = yolo(images)
            
            seg_loss = seg_criterion(outputs, labels) 
            seg_loss_rec = seg_loss[1].to('cpu') 
            
            seg_train_total_loss += seg_loss_rec

            (seg_loss[0] * 0.1).sum().backward() # segment는 0.1 가중치를 주어서 조금만 학습
            optimizer.step() 
            optimizer.zero_grad()

            current_lr = yolo_optimizer.param_groups[0]['lr']
            
            scheduler.step()
            
            wandb.log({"learning rate" : current_lr, 
                       "Detect train box loss" : det_loss_rec[0], "Detect train class loss": det_loss_rec[1], "Detect train dfl loss" : det_loss_rec[2], 
                       "Segment train box loss" : seg_loss_rec[0], "Segment train segment loss": seg_loss_rec[1], "Segment train class loss": seg_loss_rec[2], "Segment train dfl loss" : seg_loss_rec[3]
                       })
        
        with torch.no_grad(): 
            yolo.eval() 
            det_val_total_loss = torch.tensor([0., 0., 0.])
            seg_val_total_loss = torch.tensor([0., 0., 0., 0.])

            for data in tqdm(val_loader): 
                images, labels = data[0].to(device), data[1]
                    
                #---------------------------
                # Detect evaluation
                #---------------------------
                YOLOv8.mode = "detect"
                outputs = yolo(images) 
                det_loss = det_criterion(outputs, labels)
                det_val_total_loss += det_loss[1].to('cpu')
                
                #---------------------------
                # Segment evaluation
                #---------------------------
                YOLOv8.mode = "segment"
                images, labels = make_bacteria.batch_generate(batch_size = 4)
                images = images.to(device)
                outputs = yolo(images) 
                seg_loss = seg_criterion(outputs, labels)
                seg_val_total_loss += seg_loss[1].to('cpu')

                
        det_train_avg_loss = (det_train_total_loss / len(train_loader)).tolist()
        det_val_avg_loss = (det_val_total_loss / len(val_loader)).tolist()
        
        seg_train_avg_loss = (seg_train_total_loss / len(train_loader)).tolist()
        seg_val_avg_loss = (seg_val_total_loss / len(val_loader)).tolist()
        
        
        
        wandb.log({"epoch": epoch, 
                   "Detect train avg box loss": det_train_avg_loss[0], "Detect train avg class loss": det_train_avg_loss[1], "Detect train avg dfl loss": det_train_avg_loss[2],
                   "Detect validation avg box loss": det_val_avg_loss[0],"Detect validation avg class loss": det_val_avg_loss[1], "Detect validation avg dfl loss": det_val_avg_loss[2],
                   
                   "Segment train avg box loss": seg_train_avg_loss[0], "Segment train avg segment loss": seg_train_avg_loss[1], "Segment train avg class loss": seg_train_avg_loss[2], "Segment train avg dfl loss": seg_train_avg_loss[3],
                   "Segment validation avg box loss": seg_val_avg_loss[0],"Segment validation avg segment loss": seg_train_avg_loss[1],"Segment validation avg class loss": seg_val_avg_loss[2], "Segment validation avg dfl loss": seg_val_avg_loss[3]
                  })
                 
        early_stopper(det_train_avg_loss, det_val_avg_loss, yolo, epoch, det_loss_text)
            
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
          
    end = time.time() #학습이 얼마나 걸리는 지 확인 (끝)
    sec = end-start
    
    print("Training Done.")
    print(f"Elapsed Time : {int(sec)//60} minutes, {sec%60:.4f} seconds")
```


```python
learning_rate = 0.005
epochs = 16

run = wandb.init(
    entity="simple-rule-project",
    project="BYU - Locating Bacterial Flagellar Motors 2025",
     id="bj017d94",  
    resume="must",
    config={
        "epochs": epochs,
        "batch_size" : batch_size,
        "setting" : "detct & segment multi head"
    },
)

yolo_optimizer = optim.AdamW(yolo.parameters(), lr=learning_rate, weight_decay=0.01)

yolo_scheduler = OneCycleLR(
    yolo_optimizer,
    max_lr=0.0048,                # 학습률 최고점
    steps_per_epoch=len(train_loader),  # 한 에폭 안의 스텝 수
    epochs=epochs,                 # 전체 epoch 수
    pct_start=0.000001,             # 얼마나 빨리 최고점에 도달할지 (10% 지점에서 최고)
    anneal_strategy='cos',     # 감소 방식: 'cos' 또는 'linear'
    final_div_factor=1e4       # 마지막 lr은 max_lr / final_div_factor
)

yolo_early_stopper = EarlyStoppingAndCheckpoint(
    patience=5,
    model_name="Yolo",
    improvement_path='best_multi_head_trained_yolo.pt', 
    no_improvement_path='temp_multi_head_trained_yolo.pt'
)

make_bacteria = Make_bacteria_data()


```


```python
train_model(yolo, epochs, 
            det_criterion, seg_criterion, 
            yolo_optimizer,
            train_loader, val_loader, 
            yolo_early_stopper,
            yolo_scheduler,
            make_bacteria
           )

wandb.finish()
```

### ResNetMLP Training


```python
def train_model(model, epochs, 
                criterion, 
                optimizer, 
                train_loader, val_loader, 
                early_stopper, 
                scheduler
               ):
    
    start = time.time() #학습이 얼마나 걸리는 지 확인 (시작)
    
    loss_text = ["Vit loss"]
    
    for epoch in tqdm(range(epochs)):
        model.train() 
        train_total_loss = torch.tensor([0.])
        
        for data in  tqdm(train_loader): 
            data = make_z_label(data) 
            if data[0] != None:
                cropped_images, labels = data[0].to(device), data[1].to(device)
                
                outputs = model(cropped_images)
                
                loss = criterion(outputs, labels) 
                loss_rec = loss.item()
    
                loss.backward() 
                optimizer.step() 
                scheduler.step()
                optimizer.zero_grad()

                current_lr = optimizer.param_groups[0]['lr']
                
                train_total_loss += loss_rec
            
                wandb.log({"learning rate" : current_lr, 
                           "train loss": loss_rec})
        
        with torch.no_grad(): 
            model.eval() 
            val_total_loss = torch.tensor([0.])
            for data in tqdm(val_loader):       
                data = make_z_label(data)
                if data[0] != None:
                    cropped_images, labels = data[0].to(device), data[1].to(device)
                    
                    outputs = model(cropped_images)
                    loss = criterion(outputs, labels)
                    val_total_loss += loss.item()
        
        train_avg_loss = (train_total_loss / len(train_loader)).tolist()
        val_avg_loss = (val_total_loss / len(val_loader)).tolist()

        wandb.log({"epoch": epoch, 
                   "train avg loss": train_avg_loss[0], "validation avg loss": val_avg_loss[0]
                  })
                 
        early_stopper(train_avg_loss, val_avg_loss, model, epoch, loss_text)
            
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
          
    end = time.time() #학습이 얼마나 걸리는 지 확인 (끝)
    sec = end-start
    
    print("Training Done.")
    print(f"Elapsed Time : {int(sec)//60} minutes, {sec%60:.4f} seconds")
```


```python
learning_rate = 0.005
epochs = 50

run = wandb.init(
    entity="simple-rule-project",
    project="BYU - Locating Bacterial Flagellar Motors 2025",
    config={
        "epochs": epochs,
        "batch_size" : batch_size,
        "backbone": "resnet18", 
        "mlp": "4 layers", 
        "output": "11 class"
    })

resmlp_optimizer = optim.AdamW(resnet_mlp.parameters(), lr=0.005, weight_decay=0.01)

resmlp_scheduler = OneCycleLR(
    resmlp_optimizer,
    max_lr=0.005,                # 학습률 최고점
    steps_per_epoch=len(train_loader),  # 한 에폭 안의 스텝 수
    epochs=epochs,                 # 전체 epoch 수
    pct_start=0.1,             # 얼마나 빨리 최고점에 도달할지 (1% 지점에서 최고)
    anneal_strategy='cos',     # 감소 방식: 'cos' 또는 'linear'
    final_div_factor=1e4       # 마지막 lr은 max_lr / final_div_factor
)

resmlp_early_stopper = EarlyStoppingAndCheckpoint(
    patience=5,
    model_name="ResNet_MLP",
    improvement_path='best_resnet_mlp.pt', 
    no_improvement_path='temp_resnet_mlp.pt'
)
```


```python
train_model(resnet_mlp, epochs, 
            resmlp_criterion, 
            resmlp_optimizer, 
            train_loader, val_loader, 
            resmlp_early_stopper, 
            resmlp_scheduler
           )
    
wandb.finish()
```

## 6. 제출

![image](/assets/images/BYU images/submission1.png){: width="100%" height="100%"}

public 점수가 **0.25~0.3점**이었습니다(왼쪽이 대회 종료 후 공개된 Private 점수, 오른쪽이 public 점수). **baseline 점수에 크게 못미치는 점수**였습니다. 

추후 detection head로만 학습했을 때와 segmentation head를 함께 muti-head로 학습한 결과를 비교했을 때, **detection head로만 학습한 모델의 성능이 더 좋았습니다.** 그러나 점수의 향상이 미미했습니다.

실험 결과, **Multi-head 방식이 도움이 되지 않았습니다.** 

## 7. Yolo11로 재실험

마지막으로 지금까지 했던 실험에 기반하여 새롭게 모델을 학습시켜 보기로 했습니다.\
**Multi-head**로 학습하는 것이 도움이 안된다는 것을 알았으니 **Detection**만 하기로 했습니다. 그리고 yolo8 모델 대신, 라이브러리 구현체에서 제공되는 가장 최신 모델인 **yolo11**을 사용하기로 했습니다. yolo11는 **3채널** 이상 받을 수 있도록 개조하지 않고 **3채널** 입력 그대로 사용하기로 했습니다. **8장 간격**으로 **3장**을 채널 방향으로 쌓아 하나의 이미지로 만들었습니다.\
라이브러리 구현체를 그대로 사용 했을 때 내부 코드로 구현되어 있는 다양한 **augmentation**들을 쉽게 활용할 수 있다는 장점이 있습니다.\
yolo11의 모델 중 가장 사이즈가 큰 **yolo11x**를 사용했습니다.

아래가 **yolo11** 모델을 학습시키는 코드입니다.


```python
!pip install ultralytics
```


```python
from ultralytics import YOLO

model = YOLO("/kaggle/input/yolo11x/pytorch/default/4/17epochs_best.pt")

yolo_weights_dir = "/kaggle/working/yolo_weights"
os.makedirs(yolo_weights_dir, exist_ok=True)
yaml_path = "/kaggle/input/d/jobidan/byu-3channels-train-dataset/byu-3channels-train-dataset.yaml"
results = model.train(data=yaml_path, 
                      epochs=50, 
                      imgsz=640, 
                      device=[0, 1],
                      patience=6,
                      batch=16,
                      single_cls=True,
                      project=yolo_weights_dir,
                      name='motor_detector',
                      exist_ok=True,
                      lr0=0.003, # lr0 * 0.01까지 줄어듬
                      warmup_epochs=0.0001, # 스케줄러 warmup epoch 
                      cls = 0.2,
                      val = True,
                      save_period=2,
                      
                      hsv_h = 0,
                      hsv_s = 0,
                      degrees = 0.1
                     )
```

![image](/assets/images/BYU images/submission2.png){: width="100%" height="100%"}

제출 결과 점수는 **0.4**점대로 기존 **0.2**점대에서는 올랐지만 여전히 **baseline**에도 못 미치는 점수입니다.\
detection의 특성상 bbox 외의 영역이 전부 negative class에 속하기 때문에 라벨(bbox)이 존재하는 이미지 자체도 이미 **class unbalance**가 있습니다. 그런데 라벨(bbox)이 아예 없는 데이터까지 많이 학습하면 **class unbalance**가 심해집니다. baseline에서 정답 z 주변의 아주 적은 이미지만을 사용해서 학습을 했지만 그럼에도 저의 모델보다 성능이 잘 나온 이유가 **class unbalance** 때문이라고 예상합니다.

# 배운 점

이번 대회를 통해 **소통**의 중요성을 다시 한 번 깨달았습니다.\
캐글 대회에 참가한다면 개발자들이 소통하는 **Discussion**을 계속 살펴보는 것이 중요합니다. 여기에 놓치기 쉽거나 중요한 정보가 많습니다. 예를 들어, 제출할 때 데이터 타입을 integer 대신 floating point로 해야 점수가 잘 나오며, 라벨 누락 등의 문제가 있다는 이야기가 있었습니다. 이를 잘 처리하는 것도 주된 문제였습니다.

모델의 **일반화**도 매우 중요하는 것을 느꼈습니다.\
**public 점수**로 순위권 안에 들던 팀들이 대회가 끝나고 **비공개 테스트 데이터**로 **평가**되었을 때, 1등을 제외한 모든 팀이 순위권 밖으로 밀려났습니다. 1등한 참가자의 후기를 보았을 때 일반화에 노력을 많이 한 것이 보였습니다. 기업이 AI 모델을 개발할 때도 일반화가 중요할 것이라고 느꼈습니다. 

실패를 했을 때 **실패를 통해 배우는 것이 중요**하다고 느꼈습니다.\
대회에서 수상하지 못하고 "시간 낭비였나?"라는 생각을 했을 때, 실패를 하지 못했으면 발전하지 못했을 거란 생각을 했습니다. 1등한 참가자가 작성한 솔루션을 통해 그의 통찰과 전략을 배우기도 했습니다. 제가 시행착오를 겪으며 문제를 깊게 이해했기에 다른 참가자의 솔루션을 더 깊이있게 배울 수 있었다고 생각합니다.

1등 참가자의 솔루션을 공유하며 글을 마무리하겠습니다.

[1st Place Solution - 3D U-Net + Quantile Thresholding ](https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025/writeups/bartley-1st-place-3d-u-net-quantile-thresholding)


