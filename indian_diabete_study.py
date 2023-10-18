from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas 데이터 라이브러리
import pandas as pd
# matplotlib 시각화 라이브러리
import matplotlib.pyplot as plt
# seaborn 시각화 라이브러리
import seaborn as sns

df = pd.read_csv('./data/pima-indians-diabetes3.csv')

# csv 파일에 있는 데이터 행을 5개 출력
df.head(5)

# 속성별 특징을 수치화
# 1. 개수(count): 각 열에 비어있지 않은 항목의 수
# 2. 평균(mean): 각 열의 평균값
# 3. 표준편차(std) : 각 열의 표준편차
# 4. 최솟값(min) : 각 열의 최소값
# 5. 최대값(max) : 각 열의 최대값
# 6. 25%, 50% 75% 백분위 값
df.describe()

# 각 항목의 상관관계를 시각화
# 열 간의 상관 관계를 개산하여 상관 행렬을 반환(-1 ~ 1 값을 가짐)
# 1(완벽한 선형 상관 관계), 0(상관 없음), -1(완벽한 음의 선형 상관 관계)
df.corr()

# 데이터시각화에서 색상을 선택할 때 사용할 수 있는 컬러맵을 설정
# gist_heat : 높은 값에서 낮은 값으로 갈수록 어두운 색으로 표현
# cm : 데이터의 값에 따라 색상을 매핑하는 방법을 의미
colormap = plt.cm.gist_heat


# 그림의 크기가 가로 12inch * 12inch
plt.figure(figsize=(12,12))

# heatmap() : 히트맵 그래프를 그리는 함수를 호출
# dr.corr() : df의 상관 행렬을 계산
# linewidths = 0.1 : 히트맵 내의 셀(cell) 사이의 간격을 설정
# vmax = 0.5 : 상관 계수의 범위를 조절하여 시각화의 대비를 높임
# cmap=colormap : 컬러 맵 설정
# linecolor = white : 히트맵 셀 사이의 선의 색상을 설정
# annot = True : 각 셀에 숫자 값을 표시할지 여부를 설정
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)

# 라이브러리에서 그래프를 화면에 표시하는 함수
plt.show()

# X축 기준, index 0 ~ 7에 속하는 y값 전체 
X = df.iloc[:, 0:8]

# x축 기준, index 8에 속하는 y값 전체
y = df.iloc[:, 8]

# 레이어를 순차적으로 생성 및 배치
model = Sequential()

# Dense : 입력과 출력층 전체를 모두 연결
# 12, 8, 1 : 해당 레이어 층에 뉴런 또는 노드의 수
# input_dim : 입력 데이터의 특성 수
# activation : 활성화 함수
# name : 레이어 이름 부여
model.add(Dense(12, input_dim = 8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))

# 각 층의 이름(타입), 출력 모양(batch_size, 노드 개수), Param(가중치 개수)
# batch_size = none 일 때, 배치 크기를 동적으로 처리할 수 있음을 의미
model.summary()

# loss : 손실 함수, optimizer : 옵티마이저, metrics : 평가할 지표
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy'])

# model.fit() : 딥러닝 모델을 학습하는 메서드
# X : 입력 데이터의 특성
# y : 목표 출력 레이블로 모델이 예측하려는 대상 값
# epochs : 학습의 반복 횟수
# batch_size : 모델의 가중치를 업데이트하기 위한 미니 배치 크기
history = model.fit(X, y, epochs=100, batch_size=5)