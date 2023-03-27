from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch.nn.functional as F
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"]="1"

import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# mnist 데이터 불러오기
path2data = './data'
train_data = datasets.MNIST(root = path2data,
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())
valid_data = datasets.MNIST(root = path2data,
                            train=False,
                            download=True,
                            transform=transforms.ToTensor())

train_dl = DataLoader(train_data, batch_size=32, shuffle=True)
val_dl = DataLoader(valid_data, batch_size=32)

x_train, y_train = train_data.data, train_data.targets
x_val, y_val = valid_data.data, valid_data.targets

#차원 늘리기
if len(x_train.shape) == 3:
    x_train = x_train.unsqueeze(1)
if len(x_val.shape) == 3:
    x_val = x_val.unsqueeze(1)