import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class YOLO_V1(nn.Module):
    """yolov1의 논문에 나와있는 구조 구현 ( padding에 대한 이야기가 없지만 padding을 하지 않으면 에러 발생)
        kerenel_size = (7,7), st = 2 인 경우 padding = 3을 추가 하였습니다.
        kerenel_size = (3,3), st = 2 인 경우 padding = 1을 추가 하였습니다.
    """
    def __init__(self, in_channels, S, B, C):
        super().__init__()

        self.in_channels = in_channels
        self.S = S
        self.B = B
        self.C = C

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels= 64,  kernel_size=(7,7), stride=2, padding=3),
            nn.MaxPool2d((2,2), stride = 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3,3), stride= 1, padding = 1),
            nn.MaxPool2d((2,2), stride= 2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(1,1), stride=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding = 1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding = 1),
            nn.MaxPool2d((2,2), stride=2)
        )

        self.conv4_bottleNeck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1,1), stride=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding = 1),
        )
        self.conv4 = nn.Sequential(
            self.conv4_bottleNeck,
            self.conv4_bottleNeck,
            self.conv4_bottleNeck,
            self.conv4_bottleNeck,
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1,1), stride=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=1, padding = 1),
            nn.MaxPool2d((2,2), stride=2)
        )

        self.conv5_bottleNeck = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1,1), stride=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3,3), stride=1, padding = 1)
        )
        self.conv5 = nn.Sequential(
            self.conv5_bottleNeck,
            self.conv5_bottleNeck,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=1, padding=1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=2, padding=1)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=1, padding = 1),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(3,3), stride=1, padding = 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features = self.S * self.S * 1024, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features=self.S * self.S * (self.B * 5 + self.C))
        )

    def forward(self, x):
        """yolo v1 model 진행

        Args:
            x (ndarray): 448 x 448 x 3 크기의 입력

        Returns:
            tensor: S x S x (B * 5 + C) 크기의 행렬
        """
        conv1_result = self.conv1(x)
        print(f'conv1 : {conv1_result.shape}')

        conv2_result = self.conv2(conv1_result)
        print(f'conv2 : {conv2_result.shape}')

        conv3_result = self.conv3(conv2_result)  
        print(f'conv3 : {conv3_result.shape}')

        conv4_result = self.conv4(conv3_result)
        print(f'conv4 : {conv4_result.shape}')

        conv5_result = self.conv5(conv4_result)
        print(f'conv5 : {conv5_result.shape}')

        conv6_result = self.conv6(conv5_result)
        print(f'conv6 : {conv6_result.shape}')

        fc_result = self.fc(conv6_result.flatten())
        result = fc_result.view(self.S, self.S, -1)
        return result


        
def model(data, S, B, C):
    """yolo v1 모델을 구현합니다.

    Args:
        data (ndarry): 448 x 448 x 3 크기의 더미 데이터를 입력으로 받습니다.

    Returns:
        tensor: 최종 결과 tensor를 반환 합니다.
    """

    if isinstance(data, np.ndarray) : 
        data = torch.from_numpy(data).float()
    else:
        raise ValueError('Numpy 형식의 입력 데이터를 넣어 주세요.')
    

    in_channels = list(data.shape)[-1]
    yolo = YOLO_V1(in_channels, S, B, C)

    data = data.permute(2,0,1)
    result = yolo(data)
    return result