from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os

## MNIST 데이터 불러오기
# X_train : 픽셀로 이루어진 60,000개의 데이터
# y_train : 각 데이터가 가리키는 숫자
# X_test : 픽셀로 이루어진 10,000개의 데이터
# y_test : 각 데이터가 가리키는 숫자
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## 차원 변환
# 25*25로 된 배열을 1차원으로 변환
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')/255

## 원-핫코딩
# 0~9로 표시된 숫자를 0과 1로 변환
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## 모델 구조 설정
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

## 모델 실행 환경 설정
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## 모델 최적화 설정
# modelpath : 모델 저장 경로
# monitor : val_loss 수치를 모니터링
# verbose = 1 : epochs가 완료될 때마다 결과 출력
# save_best_only = True : 최적 모델 저장
modelpath = "./MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                               verbose=1, save_best_only=True)

## 학습 조기 종료
# monitor : val_loss 수치를 모니터링
# patience : 연속 10회 개선이 없으면
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

## 모델 실행
# X_train, y_train을 바탕으로 학습
# validation_split = 0.25 : 학습셋의 25%
# epochs = 30 : 전체 데이터를 30번 반복
# batch_size = 200 : 샘플 200개 학습 후 가중치 업데이트
# verbose = 0 : 결과 출력 X
# callbacks : 호출할 함수 조기 종료 및 모델 저장
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30,
                    batch_size=200, verbose=0, callbacks=[early_stopping_callback,
                                                          checkpointer])

## 테스트 정확도 출력
# model.evaluate의 결과 [loss, accuracy]
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

## 검증셋과 학습셋의 오차를 저장
# history의 결과 [loss, accuracy, val_loss, val_accuracy]
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

## 그래프 표현
# x축의 길이는 len(y_loss)만큼의 길이로 설정
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")

## 그래프에 그리드를 주고 레이블을 표시
# 범례 위치 지정
plt.legend(loc='upper right')
# 가로선과 세로선 그리드 표시(비교 용이)
plt.grid()
# x축 범례 지정
plt.xlabel('epoch')
# y축 범례 지정
plt.ylabel('loss')
plt.show()