# tensorlow 패키지 묶음으로 상위 package임
# keras는 모듈 묶음으로 package라 부르며, 
# 상위 package + package는 C++에서 라이브러리로 비유할 수 있음
# moels, layers는 모듈이라 부르며, C++에서 분할 컴파일 하기 위한 소스 파일에 비유할 수 있음
# import Sequential, Dense는 모듈에 있는 함수를 의미하며,
# C++에서 소스 파일에 작성된 함수에 비유할 수 있음
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 라이브러리에 포함된 numpy란 함수를 사용하겠다는 의미임
# include<iostream> 의 cout = numpy
# 불러온 함수 numpy를 np라는 이름으로 사용하겠다는 의미임
import numpy as np

# 깃허브에 준비된 데이터를 가져옴
!git clone https:#github.com/taehojo/data.git

# ThoraricSurgery3.csv 파일을 불러와서 "," 구분자로 numpy 배열에 저장됨
Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimter=",")
# 훈련 데이터셋: 모델을 학습하는데 사용되는 데이터
# 검증 데이터셋: 모델의 성능을 평가하고 하이퍼파라미터를 조정하는데 사용되는 데이터
# 테스트 데이터셋: 최종 모델의 성능을 평가하는데 사용되는 데이터

# X축 기준 속성 인덱스 0부터 15까지의 y축 전체 데이터인 진찰 기록을 담음
X = Data_set[:, 0:16]
# X축 기준 인덱스 16(사망/생존 여부)의 y축 전체 데이터를 지정
y = Data_set[:, 16]

# 모델의 종류
# 1. Sequential : model.add로 생성한 레이어를 순차적으로 쌓음.
# 2. Functional API: 레이어를 보다 복잡한 구조로 연결할 수 있어 1층에 1개, 2층에 2개를 두고
#                    1층에서 2층의 2개의 Layer에 연결할 수 있음.
# 3. 서브클래싱 모델: 사용자 지정 모델을 만들 때, 유리함
model = Sequential()


# model.add: 레이어를 추가
# model.add(레이어 종류(units, input_dim = , activation = ))

# 레이어의 종류
# Dense(Fully Connected) 레이어: 각 뉴런이 이전 레이어의 모든 뉴런과 연결되는 레이어
# Convolutional 레이어: 합성곱 연산을 사용하여 이미지와 같은 데이터에서 특징을 추출
# Recurrent 레이어: 순차 데이터에 대한 모델링에 사용되며, LSTM 및 GRU와 같은 변형이 존재
# Dropout 레이어: 과적합을 방지하기 위해 사용되어, 일부 뉴런을 무작위로 비활성화

# units: 해당 레이어가 가질 뉴런 혹은 유닛의 수를 결정
# input_dim: 입력 데이터의 차원을 정의하는 매개변수로 입력 데이터의 특성 수나 요소의 개수를 의미

# activation: 활성화 함수
# ReLU(Rectified Linear Unit): 주로 은닉 레이어에서 사용되며, 비선형을 도입
# Sigmoid: 이진 분류 문제에서 출력 레이어에 사용
# Softmax: 다중 클래스 분류 문제에서 출력 레이어에 사용, 확률 분포를 생성
# tanh: 종종 순환 신경망(RNN)에서 사용하고 -1과 1사이의 값을 출력
model.add(Dense(30, input_dim = 16, activation='relu'))
model.add(Dense(1, activation='sigmold'))

# model.compile: 모델 컴파일
# model.compile(loss =, optimizer =, metrices =[])
# loss: 손실 함수
# Binary Cross-Entropy: 이진 분류 문제에 사용
# Categorical Cross-Entropy: 다중 클래스 분류 문제에 사용
# Mean Squared Error(평균 제곱 오차): 회귀 문제에 사용
# Hinge Loss: SVM과 같은 분류 문제에 사용

# optimizer: 옵티마이저
# Adam: 최적화 알고리즘
# Stochastic Gradient Descent(SGD): 오래된 최적화 알고리즘, 확률적 경사 하강법
# RMSprop: Adgrad의 변형으로 학습률을 조절하는 방식이 개선

# Metrices: 모델의 성능을 평가하기 위한 지표
# 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수(F1 Sceore)
# 회귀 모델의 경우, 평균 제곱 오차, 평균 절대 오차 등이 존재
model.compile(loss='binary_crossentropy', optimizer='adam', metrices =['accuracy']);



# history에 담기는 정보
# 학습 과정에서 발생한 정보를 저장하며, 학습 중 관찰된 손실(loss) 및 성능 지표(metrics)

# model.fit: 컴파일 된 모델 실행
# X: 입력 데이터로 모델이 학습할 데이터
# y: 출력 데이터로 X에 대한 예측을 만들고 y와 비교한 후,
#    손실을 계산하여 역전파 알고리즘으로 가중치 업데이트
# Epochs: 데이터셋 전체를 학습하는 횟수
# Batch Size: n개의 학습을 진행한 후, 모델 가중치 업데이트를 수행
history = model.fit(X, y, epochs = 5, batch_size = 16)