# Introduction

yolo v1의 장점.

1. End to End pipline의 아키텍처로 다른 모델들 보다 빠른 inference 시간을 갖는다.
2. 전체 이미지를 input으로 사용하기에 이미지의 문맥정보를 잘 파악한다.
3. 일반화 성능이 다른 모델에 비해 높다. ( 예술작품에 대한 성능이 높다 ) 

그래도 아직 다른 최첨단 모델들에 비해서는 정확도가 떨어지는 측면이 존재한다.

## Unified Detection

yolo v1의 경우 이미지에서 객체로 판단되는 모든 박스를 동시에 검출합니다. 
우선 이미지를 S X S 크기의 grid로 분할합니다. 그리고 각 grid cell 내부에 객체의 center 좌표가 포함되게 되면 해당 cell이 해당 객체를 판별하는 역할을 수행합니다. 각 grid cell은 B개의 박스를 생성하게 됩니다. 각 cell은 $(x, y, w, h, score)$ 총 5개의 값을 갖게 됩니다. (x,y)는 실제 객체의 중심 좌표를 나타내며, (w,h)는 실제 이미지의 크기를 나타냅니다, 그리고 score의 경우 실제 bounding box와의 IOU 점수를 나타내게 됩니다. 그리고 각 셀들은 C개의 조건부 확률을 예측하게 됩니다. 이는 $Pr(Class_i | Object )$ 즉 해당 객체가 i 번쨰 클래스에 속할 확률을 의미하게 됩니다. score의 경우 $Pr(Class_i | Object ) * IOU(truthpred)$ 로 사용된다고 합니다. ( 객체가 있을 확률  x IOU점수 )

최종적으로 ( S, S, B * 5 + C ) 의 tensor를 갖게 됩니다. B개의 박스 마다 5개의 값인 $(x, y, w, h, score)$을 예측하고, 각 cell이 어느 class에 속할지를 결정하는 조건부 확률을 예측해야하기 떄문입니다. 

## Training

우선 ImageNet 데이터셋을 활용해서 사전학습 시켰습니다. 사전 학습을 시키기 위해서 20개의 CONV Layer와 average pooling + FC layer를 사용해서 학습 시켰습니다. 1주일 가량 학습 시킨 모델의 성능은 당시 ImageNet에서 5등을 기록했다고 합니다. 이후 추가적인 성능 개선을 위해서 4개의 conv layer와 2개의 fc layer를 사용한 모델을 만들었다고 합니다. 그리고 모델이 이미지의 세부적인 패턴을 파악하기 위해서 (224, 224) 이미지 크기에서 (448, 448) 크기의 이미지를 사용했다고 합니다. 
최종적으로 모델은 박스의 위치와 해당 박스가 속할 클래스의 확률을 출력하였습니다. 그리고 박스의 위치는 모든 값이 0 ~ 1사이의 값으로 정규화 되어나온다고 합니다. 그리고 마지막 Layer에서 활성화 함수를 사용하는데, 이때 는 Leaky ReLU 함수를 사용한다고 합니다.

$$
\phi(x) ={ x, \ \ \ \   \text{if x > 0} \brace 0.1x  \ \ \ \ \text{otherwise}}
$$

그리고 Loss function으로는 sum-squared loss를 사용하게 되면 classification loss와 localization loss를 동등하게 부여하기에, 기본적으로 객체를 포함하지 않은 cell들이 많아 clasification loss가 0이되는 경우가 다수 존재한다고 합니디. 이로 인해 초반에 불안정하게 학습이 진행될수 있다고 합니다.

그래서 $\lambda_{coord}$와 $\lambda_{noobj}$을 두어 객체를 포함하는 cell과 객체를 포함하지 않는 cell의 가중치를 다르게 설정하여 위의 문제를 해결하고자 하였습니다. 논문에서는 $\lambda_{coord} = 5$ $\lambda_{noobj} = 0.5$ 로 설정하였습니다.
그리고 크기가 큰 박스의 작은 오차가 작은 박스의 작은 오차 보다 덜 중요합니다. 즉 박스 크기에 따른 오차를 어느정도 보완하기 위해서 차이를 구할 시 $(\sqrt{w_i} - \sqrt{\hat{w_i}})^2$ 와 $(\sqrt{h_i} - \sqrt{\hat{h_i}})^2$ 같이 제곱근의 차이를 구했다고 합니다.
그리고 yolo v1 모델에서는 하나의 cell에서 B개의 박스를 생성하게 됩니다. 하지만 우리는 이중 하나의 박스만을 사용하여 예측을 하기 위해서 대표 박스를 선정하게 됩니다. 가장 높은 IOU를 갖는 박스가 대표 박스가 됩니다. 그래서 각 예측기는 특정 크기, 종횡비 또는 객체 클래스에 대해 더 예측을 하게 되어 자연스럽게 전체적인 recall을 향상 시키게 된다고 합니다.

** Precision → $TP \over TP + FP$ 즉, 검출한 결과가 얼마나 정확한가? **

** Recall → $TP \over TP + FN$ 즉, 실제 정답들을 빠트리지 않고 잘 잡아 냈는가? **

그래서 yolo v1의 최종적인 loss function은 아래와 같습니다.

$$
\lambda_{coored}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j} [ ({x_i} - {\hat{x_i}})^2 + ({y_i} - {\hat{y_i}})^2]  \ \ \ \ + \lambda_{coored}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j} [ (\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h_i}})^2] + \sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j}(C_i - \hat{C_i})^2 + \lambda_{nobbj}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j}(C_i - \hat{C_i})^2 + \sum^{s^2}_{c \in classes} (p_i(c) - \hat{p_i}(c))^2
$$

위의 수식에 대해 천천히 분석을 해보면 다음과 같습니다.

$1^{obj}_{i,j}$ : cell i에 객체 여부를 나타내며, 각 cell의 j 번쨰 박스가 대표 박스임을 나타냅니다.
Localization Loss는 객체가 존재하는 대표 박스에 대해서만 손실을 부여합니다.

### Localization Loss

$\lambda_{coored}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j} [ ({x_i} - {\hat{x_i}})^2 + ({y_i} - {\hat{y_i}})^2]$ : 박스 중심 좌표에 대한 손실 함수

$\lambda_{coored}\sum^{S^2}_{i=0}\sum^B_{j=1}1^{obj}_{i,j} [ (\sqrt{w_i} - \sqrt{\hat{w_i}})^2 + (\sqrt{h_i} - \sqrt{\hat{h_i}})^2]$ : 박스 크기에 대한 손실 함수
이의 식에서 모두 큰 박스와 작은 박스의 크기 조정을 위해서 제곱근을 사용하여 표현하였습니다..

### Confidence Loss

$\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j}(C_i - \hat{C_i})^2$  : 객체인 박스에 대한 confidence 손실 함수 ( 실제 클래스의 index 값은 1 이다 )

$\lambda_{nobbj}\sum^{S^2}_{i=0}\sum^B_{j=0}1^{obj}_{i,j}(C_i - \hat{C_i})^2$ : 객체가 없는 box에 대한 confidence 손실 함수

### Classification Loss

 $\sum^{s^2}_{c \in classes} (p_i(c) - \hat{p_i}(c))^2$ : 클래스에 대한 분류 손실 함수
