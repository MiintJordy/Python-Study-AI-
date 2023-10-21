""" 컨볼루션 신경망 """
"""
컨볼루션 신경망은 입력된 이미지에서 다시 한 번 특징을 추출하기 위해 커널(슬라이딩 윈도)을 도입하는 기법임.

@ 슬라이딩 윈도란?
- 컴퓨터 비전 및 신호 처리와 같은 영역에서 사용되는 일반적인 기술임. 
이 기술은 데이터를 분석하거나 처리할 때 데이터의 작은 부분을 순차적으로 이동시키면서 작업을 수행함.

4*4의 크기로 0과 1로 이루어진 이미지가 있다면,
2*2 크기의 X0 X1의 가중치로 설정된 커널을 만들고,
한 칸씩 이동하며 순차적으로 겹치면 9개의 결과가 나옴

3*3으로 나온 새롭게 만들어진 층을 컨볼루션(합성곱) 층이라고 하며, 
컨볼루션 층을 만들면 입력 데이터가 가진 특징을 대략적으로 추출해서 학습을 진행할 수 있음.

# keras에서 컨볼루션 층을 추가하는 함수 : Conv2D()
# 32 : 32개의 커널을 사용
# kernel_size(3, 3) : 커널 크기는 3*3
# input_shape(28, 28, 1) : 28행, 28열, 1 흑백  // 색상이 다양하면 3 
model.add(Conv2D(32, kernel_size(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))


위에서 구현한 컨볼루션 층을 통해 이미지 특징을 도출하였음. 
하지만, 그 결과가 여전히 크고 복잡하면 이를 다시 한 번 축소해야함. 이 과정을 풀링(pooling) 
또는 서브 샘플링(sub sampling)이라고 함.

이러한 풀링 기법에는 정해진 구역 안에서 최대값을 뽑아내는 
맥스 풀링(Max Pooling)과 평균 값을 뽑아내는 평균 풀링(Average Pooling)이 있음.

4*4의 표에서 2*2의 크기로 구역을 4개로 나누고 각 구역에서 가장 큰 값을 추출하여 불필요한 정보를 간추림.

맥스풀링은 아래와 같은 함수를 사용해서 다음과 같이 적용할 수 있음.

# (2,2)는 가로 2, 세로 2 크기의 풀링창을 통해 맥스 풀링을 진행하라는 의미임.
model.add(MaxPooling2D(Pool_size(2,2)))

노드가 많아지거나 층이 많아진다고 해서 학습이 무조건 좋아지는 것이 아니기 때문에, 
과적합을 얼마나 효과적으로 피해 가는지가 매우 중요하여 이에 대한 기법이 연구되어 왔음. 
그중 간단하지만 효과가 큰 기법이 "드롭 아웃(Drop Out) 기법"임. 
드롭아웃은 은닉층에 배치된 노드 중 일부를 임의로 꺼주는 것임.
랜덤하게 노드를 꺼주면 학습 데이터에 지나치게 치우쳐서 학습되는 과적합을 방지할 수 있음.

# 25의 노드를 끔
model.add(Dropout(0.25)

Dense() 함수를 이용해 만들었던 기본 층에 연결할 때, 
주의할 점은 컨볼루션 층이나 맥스 풀링은 주어진 이미지를 2차원 배열인 채로 다룬다는 것임. 
이를 1차원 배열로 바꾸어 주어야 활성화 함수가 있는 층에서 사용할 수 있음. 
따라서, Faltten() 함수를 사용해 2차원 배열을 1차원으로 바꾸어줌

레이어에 사용되기 위해서는 1차원 배열이 되어야 함.
model.add(Flatten())

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# input_shape : 28*28 크기의 흑백 이미지
# Conv2D(32, kernel_size(3,3)) : 3*3 크기의 32개의 커널을 사용
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))

# 3*3 크기의 커널 62개를 사용
model.add(Conv2D(62, (3,3), activation='relu'))

# 2*2 크기의 커널로 맥스풀링 적용
model.add(MaxPooling2D(pool_size=(2,2)))

# 노드 25% 끄기
model.add(Dropout(0.25))

# Dense층에 연결하기 위해 다차원 데이터를 1차원 데이터로 변환
model.add(Flatten())

# 128개 노드 생성
model.add(Dense(128, activation='relu'))

# 노드 50% 끄기
model.add(Dropout(0.5))

# 10개 노드 생성
model.add(Dense(10, activation='softmax'))

# softmax니까 손실함수는 categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 최적화
modelpath = "./MNIST_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose = 1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 10)

# 모델 실행
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, y_test)[1]))

# 검증 셋과 학습셋의 오차 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker=".", c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()