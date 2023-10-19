# 모델 과적합을 그래프로 확인

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd
""" 그래프 관련 함수 추가 """
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/wine.csv', header = None)

# 데이터 보기
# df

# X에 와인의 속성, y에 와인의 분류
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# 학습셋과 테스트셋으로 분류
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer ='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=500,
                    validation_split=0.25)

# history에 저장된 학습 결과를 확인
hist_df = pd.DataFrame(history.history)
# loss accuracy val_loss val_accuracy를 속성으로 epochs번의 학습 결과가 저장
hist_df

# val_loss를 y_vloss에 저장
y_vloss = hist_df['val_loss']

# loss를 y_loss에 저장
y_loss = hist_df['loss']

# x축 값으로 사용할 배열을 생성
x_len = np.arange(len(y_loss))
# 검증 데이터의 위치 : (x_len, y_vloss)
# 검증 데이터의 표시 : "o"(원), c (색깔), markersize(크기), label(범례)
# s(사각형), ^(삼각형), d(다이아몬드), x(엑스), *(별표)
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Testset_loss')
# 훈련 데이터의 위치 : (x_len, y_loss)
# 훈련 데이터의 표시 : "o"(원), c (색깔), markersize(크기), label(범례)
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trainset_loss')

# 범례의 위치를 오른쪽 상단
# upper left, lower right, lower left
plt.legend(loc='upper right')
# x축 범례
plt.xlabel('epoch')
# y축 범례
plt.ylabel('loss')
plt.show()