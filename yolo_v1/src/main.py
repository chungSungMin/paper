import numpy as np
from model import model


if __name__ == '__main__' :
    data = np.random.randint(0, 256, size=(448, 448, 3))
    print(f'입력 이미지 사이즈 : {data.shape}')

    S = 7
    B = 2
    C = 20

    result = model(data,S, B, C)
    print(f'yolo result : {result.shape}')
