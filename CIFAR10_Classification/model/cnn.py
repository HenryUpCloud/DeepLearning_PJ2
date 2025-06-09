'''
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu4(self.fc1(x)))
        return self.fc2(x)

    def extract_features(self, x):  # 用于提取特征向量
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        return x
'''

# Wider CNN
import torch
import torch.nn as nn

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class CNN(nn.Module):
    def __init__(self, dropout=0.5, activation_name="relu"):
        super(CNN, self).__init__()
        act = get_activation(activation_name)

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = act
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = act

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = act

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.act4 = act
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = self.pool(self.act3(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        x = self.dropout(self.act4(self.fc1(x)))
        return self.fc2(x)

    def extract_features(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = self.pool(self.act3(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 4 * 4)
        return x
