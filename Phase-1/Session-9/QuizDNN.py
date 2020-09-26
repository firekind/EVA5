import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        dropout_value = 0.1
        super().__init__()
        self.x2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )
        self.x3 = nn.Sequential(
            nn.Conv2d(19, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )
        self.x4 = nn.MaxPool2d(2, 2)

        self.x5 = nn.Sequential(
            nn.Conv2d(51, 16, kernel_size=1),
        )
        self.x6 = nn.Sequential(
            nn.Conv2d(67, 67, kernel_size=3, padding=1),
            nn.BatchNorm2d(67),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )
        self.x7 = nn.Sequential(
            nn.Conv2d(134, 134, kernel_size=3, padding=1),
            nn.BatchNorm2d(134),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )
        self.x8 = nn.MaxPool2d(2, 2)

        self.x9 = nn.Sequential(
            nn.Conv2d(217, 64, kernel_size=1),
        )
        self.x10 = nn.Sequential(
            nn.Conv2d(281, 281, kernel_size=3, padding=1),
            nn.BatchNorm2d(281),
            nn.ReLU(),
            nn.Dropout2d(dropout_value)
        )
        self.x11 = nn.Sequential(
            nn.Conv2d(562, 128, kernel_size=1),
        )

        self.x12 = nn.AdaptiveAvgPool2d((1, 1))
        self.x13 = nn.Linear(128, 10)

    def forward(self, x):
        o_x2 = self.x2(x)
        o_x3 = self.x3(torch.cat([x, o_x2], dim=1))
        o_x4 = self.x4(torch.cat([x, o_x2, o_x3], dim=1)) # 54

        o_x5 = self.x5(o_x4) # 16
        o_x6 = self.x6(torch.cat([o_x4, o_x5], dim=1)) # 70
        o_x7 = self.x7(torch.cat([o_x4, o_x5, o_x6], dim=1)) # 140
        o_x8 = self.x8(torch.cat([o_x5, o_x6, o_x7], dim=1)) # 226

        o_x9 = self.x9(o_x8) # 64
        o_x10 = self.x10(torch.cat([o_x8, o_x9], dim=1)) # 290
        o_x11 = self.x11(torch.cat([o_x8, o_x9, o_x10], dim=1)) # 580

        o_x12 = self.x12(o_x11)
        o_x13 = self.x13(o_x12.view(-1, 128))
        
        return F.log_softmax(o_x13, dim=-1)


        