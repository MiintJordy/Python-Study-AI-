# MNIST 손으로 쓴 숫자(0~9)를 포함하는 이미지 데이터셋
# 기계학습과 딥 러닝 모델을 학습 및 테스트하기 위한 범용적인 데이터셋
from tensorflow.keras.datasets import mnist
# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical
# 데이터 시각화
import matplotlib.pyplot as plt
# 명령 라인에서 스크립으를 실행할 때 전달하는 인수를 처리하거나 실행 환경 정보를 얻을 때 사용
import sys

# (a, b) : 괄호로 둘러싼 표현은 Python에서 튜플을 나타냄
# 데이터를 효과적으로 분할하면서 가독성을 높이기 위해서임
# 학습에 사용될 부분: X_train, y_train
# 테스트에 사용될 부분: X_test, y_test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape[0] : 속성의 개수
# shape[1] : 열의 개수
# print("학습셋 이미지 수: %d" % (X_train.shape[0]))
# print("테스트셋 이미지 수: %d" % (X_test.shape[0]))

# 불러온 이미지 중 한 개를 불러와서 확인해보기
# plt.imshow(X_train[0], cmap='Greys')
# plt.show()

# 아래 코드로 글자를 0~255까지 등급을 매긴 것을 확인
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write("%-3s" % i)
#     sys.stdout.write('\n')

# 주어진 가로 28, 세로 28의 2차원 배열을 784개의 1차원 배열로 바꿈
X_train = X_train.reshape(X_train.shape[0], 784)

# keras는 데이터를 0에서 1사이의 값으로 변환한 후 구동할 때 최적의 성능을 보임
# 데이터 정규화 : 데이터의 폭이 클 때 적절한 값으로 분산의 정도를 바꾸는 과정
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64')/255

# y_train : array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
# print("class : %d" % (y_train[0]))

# 딥러닝의 분류문제를 해결하기 위해서는 원-핫 인코딩 방식을 적용해야함
# 0~9의 정수형 값을 갖는 현재 형태에서 0 또는 1로만 이루어진 벡터로 값을 수정해야함
# to_categorical(label, num) : 해당 label의 개수는 num개란 의미
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# array([[0., 0., 0., ..., 0., 0., 0.],
#        [1., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 1., 0.]], dtype=float32)
#y_train