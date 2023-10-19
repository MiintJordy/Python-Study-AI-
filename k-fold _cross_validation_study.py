from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# scikit-learn 라이브러리에서 제공하는 KFole 사용(교차 검증 기능)
from sklearn.model_selection import KFold

# sckikkit-learn 라이브러리에서 제공하는 accuracy_score(성능 평가)
# 예측 결과 [0, 1, 1, 0, 1, 1, 0, 1]
# 실제 레이블 [0, 1, 0, 0, 1, 1, 1, 1]
# 정확도 계산 accuracy = accuracy_score(예측 결과, 실제 레이블)
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv('./data/sonar3.csv', header=None)

X = df.iloc[:, 0:60]
y = df.iloc[:, 60]

# 겹의 개수
k = 5

# 분할하기 전에 샘플이 치우치지 않도록 섞음
# 5개의 fold로 나누어진 후, 5번의 검증 과정이 진행
kfold = KFold(n_splits=k, shuffle=True)

# 정확도가 채워질 빈 리스트를 준비
acc_score = []

# 딥러닝 모델을 생성하는 함수
def model_fn():
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

# K겹 교차 검증을 이용해 K번의 학습을 실행
# for문에 의해 K번 반복
# split()에 의해 K개의 학습셋, 테스트셋으로 분리

# kfold.split(X)를 호출할 때, Line 24의 n_splits=k로 사용되므로 5등분
# 학습셋과 테스트셋을 5등분하고 5번 반복하겠다는 의미
for train_index, test_index in kfold.split(X):

    # 학습셋과 테스트셋에 값을 초기화
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 딥러닝 모델 값을 model에 초기화
    model = model_fn()
    # 모델 컴파일
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    # 학습 과정 중 발생하는 정보가 딕셔너리 형태로 history에 저장
    # loss 손실 값, accuracy 정확도 값, val_loss 각 학습 epochs에 대한 손실값, 
    # val_accuracy 각 학습 epochs에서 검증 데이터에 대한 정확도 값
    # 5등분 한 데이터셋을 셋 당 100번씩 반복하여 500회가 실행
    # verbose = 0, 학습 과정에서 어떠한 출력도 표시되지 않음
    # verbose = 1, 1등분의 데이터 셋이 100번이 반복된 후 정보 출력
    # verbose = 2, 모든 반복에 출력되어 총 500회가 출력됨
    history = model.fit(X_train, y_train, epochs = 100, batch_size=10, verbose=0)

    # 테스터 데이터를 사용하여 학습된 모델을 평가하고 평가 지표를 반환하는 메서드
    # 주로 손실과 지정한 메트릭을 반환하여 [loss, accuracy]의 구조임.
    accuracy=model.evaluate(X_test, y_test)[1]
    # 1번 인덱스의 정확도를 리스트에 append
    acc_score.append(accuracy)

# 평균 정확도 = 정확도의 합 / 5
avg_acc_score = sum(acc_score) / k
print('정확도:', acc_score)
print('정확도 평균:', avg_acc_score)