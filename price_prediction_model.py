# 레이어 순차적으로
from tensorflow.keras.models import Sequential
# 모든 노드 연결
from tensorflow.keras.layers import Dense
# 조기 종료 라이브러리
from tensorflow.keras.callbacks import EarlyStopping
# 테스트셋 라이브러리
from sklearn.model_selection import train_test_split
# 체크포인트 라이브러리
from tensorflow.keras.callbacks import ModelCheckpoint

# 그래프 출력
import matplotlib.pyplot as plt
# 시각화 및 통계 도구
import seaborn as sns
# 다차원 배열과 수학 함수
import pandas as pd
# 데이터 조작과 분석
import numpy as np
# 데이터 시각화
import matplotlib.pyplot as plt

# 해당 경로에 있는 파일의 데이터 담기
df = pd.read_csv("./data/house_train.csv")
# 카테고리형 변수를 숫자로 변경
df = pd.get_dummies(df)

# 결손 데이터를 해당 열의 평균값으로 저장
df = df.fillna(df.mean())
# 데이터 사이의 상관관계를 저장
df_corr = df.corr()
# 집 값과 관련이 큰 것 순으로 저장
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)
# 정렬된 열에서 SalePrice를 제외하고 상위 5개의 열 저장
cols_train = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
# cols_train에 속하는 리스트의 속성 값을 X_train_pre에 저장
X_train_pre = df[cols_train]
# SalePrice의 값을 저장
y = df['SalePrice'].values

# 전체의 80%를 학습셋, 20%를 테스트셋으로 지정
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

# 모델 설정
model = Sequential()
# X_train.shape[1] : X_train에 있는 속성의 개수
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

# 모델 실행 (새로운 손실 함수: 회귀에 주로 사용)
model.compile(optimizer='adam', loss='mean_squared_error')

# 20번 이상 결과가 향상되지 않으면 자동으로 중단
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# 모델 저장 경로
modelpath = "./data/model/Ch15-house.hdf5"

# 모델 저장 조건(파일 경로, 포커스: 검증데이터 손실, 정보 출력 X, 제일 높을 때)
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# 모델 실행
# X_train: 입력 특성, y_train: 목표 변수
# validation_split = 0.25, 학습 데이터의 25%를 검증 데이터로 사용
# epochs = 2000(회 반복), batch_size=32(미니배치 단위로 가중치 업데이트)
# callbacks : 모델 학습 주에 호출되는 콜백 함수의 리스트
# early_stopping_callback: 조기 종료
# checkpointer: 모델 가중치를 저장하는 콜백으로 주어진 조건을 충족하는 경우 저장
# X_train, y_train 을 사용하여 학습
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32,
                    callbacks=[early_stopping_callback, checkpointer])

# 학습 결과 시각화
# 실제 집 값을 담을 리스트
real_prices = []
# 예측 집 값을 담을 리스트
pred_prices = []
# 횟수를 담을 리스트
X_num = []
#시작 횟수는 0
n_iter = 0

# model.predict(X_test) : X_test 모델을 사용하여 X_test에 대한 예측 값을 계산하는 부분
# flatten() : 다차원 배열을 1차원으로 펼치는 역할, 예측 결과를 1차원 배열로 얻을 수 있음
Y_prediction = model.predict(X_test).flatten()

# 25회 반복
for i in range(25):
    # 실제 가격을 y_test에서 가져와서 real 변수에 저장
    real = y_test[i]
    # 예상 가격을 Y_prediction에서 가져와서 prediction 변수에 저장
    prediction = Y_prediction[i]
    # 실제 가격과 예상 가격을 소수점 2자리 까지 출력
    print("실제 가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    # 실제 가격을 리스트에 append
    real_prices.append(real)
    # 예상 가격을 리스트에 append
    pred_prices.append(prediction)
    # 반복 횟수를 증가
    n_iter = n_iter + 1
    # 횟수를 리스트에 append
    X_num.append(n_iter)

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
# 범례 추가
plt.legend()
plt.show()