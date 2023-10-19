from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

""" 모델 업데이트 """
# 학습 중인 모델을 저장하는 함수
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd

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

""" 모델 업데이트 """
# 50번째 epoch의 검증셋정확도가 0.9346이라면 50-0.9346.hdf5라는 이름으로 저장됨
modelpath = "./data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5"

# 모델 가중치를 저장할 파일 경로를 정하고 진행 현황을 모니터링
checkpointer = ModelCheckpoint(filepath=modelpath, verbose = 1)
""" """

# test set을 설정했을 때의 코드
# history = model.fit(X, y, epochs = 50, batch_size = 500)
# 검증셋 추가 시, validation_split이 추가, 학습셋에서의 0.25%로 0.8*0.25 = 0.2, 20%임
# 검증셋 역시 테스트 셋과 마찬가지로 모니터링에 사용
""" 모델 업데이트
callbacks = [checkpointer]
학습 중에 주기적으로 모델 가중치를 저장하는 함수를 부름

ModelCheckpiint : 모델의 가중치를 주기적으로 저장하거나 최적 모델을 선택하여 저장
EarlyStoppint : 지정된 조건이 충족될 때 학습을 조기 중단
ReduceLROnPlateau : 학습률을 동적으로 조절하여 학습을 최적화
사용자 정의 콜백
"""
history = model.fit(X_train, y_train, epochs=50, batch_size=500,
                    validation_split=0.25, verbose = 0, callbacks=[checkpointer])

score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])