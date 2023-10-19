from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas 데이터 라이브러리
import pandas as pd

# seaborn 시각화 라이브러리
import seaborn as sns

# matplotlib 시각화 라이브러리
import matplotlib.pyplot as plt
df = pd.read_csv('./data/iris3.csv')

# 숫자 미입력시, 데이터 상위 5개 출력
# df.tail() : 숫자 미입력시, 데이터 하위 5개 출력
# df.head()

# sns.pairplot() : 데이터프레임의 모든 숫자형 열에 대해 산점도 행렬을 생성
# df : 산점도 행렬을 그릴 데이터 프레임
# hue = 'speices' : hue 매개변수는 데이터를 그룹화하거나 색상을 구분하는데 사용
# speices : 해당 열은 각 꽃 데이터가 어떤 품종에 속하는지 나타내는 열
# 특정 그래프만 출력할 때, sns.pairplot(df, hue='species', vars=['sepal_length'])
#sns.pairplot(df, hue='species')
#plt.show()

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# 출력하였을 때, y의 값이 숫자가 아닌 문자로 저장되어있어 숫자형으로 변환이 필요
# print(X[0:5])
# print(y[0:5])

# 품종 열에 대해서 각각 범주를 나타내는 세 개의 열로 변환
# (원-핫 인코딩 처리)
# [1, 0, 0] [0, 1, 0] [0, 0, 1]의 형태로 원은 1, 핫은 활성화를 의미
y = pd.get_dummies(y)
# print(y[0:5])

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
# softmax
# sigmoid의 경우, 0부터 1까지 사이 수를 반환하고 0.5이하는 ~, 이상은 ~
# 하지만, 이 코드의 경우 품종 3개 중에 선택해야 하므로 softmax를 사용
# softmax는 0부터 1 사이의 값을 반환하고 각 샘플당 예측 확률의 총 합이 1인 형태로 변환
# 예를 들어 0.2 + 0.7 + 0.1 = 1
model.add(Dense(3, activation='softmax'))
# 생성한 모델의 요약 정보: 레이어 이름(타입), 각 층의 출력 모양, 레이어 파라미터 수
# model.summary()

# 그래서 sigmoid는 이항 분류로 손실 함수를 이항 분류 binary_crossentropy를 사용
# softmax는 다항 분류로 categorical_crossentropy를 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 실행
histroy = model.fit(X, y, epochs=50, batch_size=5)
