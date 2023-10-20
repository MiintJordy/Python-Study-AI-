# 프로그램 조기 조욜

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 라이브러리에서 제공되는 콜백 중 하나로 모델의 학습을 조기에 중단
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/wine.csv', header = None)


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

# monitor = val_loss : 모니터링할 지표 설정(이 코드에서는 검증 데이터의 손실)
# patience : 지정된 지표가 계선되지 않는 epoch 횟수 지정
# 예를 들어 20으로 설정했으면 연속 20번동안 개선되지 않으면 학습을 조기 중단
# verbose : 중단 조건이 충족되었을 때 출력할 메시지의 상세도를 지정
# mode : 모니터링할 지표의 개선 방향을 지정(min 손실이 낮은 좋은 지표, max 정확도와 같이 높을수록 좋은 지표)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 20)

modelpath = "./data/model/Ch14-4-bestmodel.hdf5"

# monitor = val_loss : 검증 데이터의 손실을 모니터링하여 가장 좋은 모델을 저장하도록 설정
# save_best_only : 가장 좋은 성능을 갖는 모델만 저장할 것인지 여부
# 매 epoch를 반복하면서 가장 가중치가 높은 것을 메모리에 저장하고 있다가 높은 것이 나오면 갱신
# 프로그램 종료 시에 가장 높았던 것을 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(X_train, y_train, epochs = 2000, batch_size =500, validation_split = 0.25, verbose = 1,
                    callbacks = [early_stopping_callback, checkpointer])