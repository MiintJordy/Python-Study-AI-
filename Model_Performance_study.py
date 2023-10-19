from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

"""
sklearn 라이브러리에서 제공되는 함수로 학습용과 테스트용을 나눔
이 함수는 데이터셋을 무작위로 섞고 지정된 비율에 따라 
학습 데이터와 테스트 데이터로 나눔
"""
from sklearn.model_selection import train_test_split

#

# pandas 데이터 라이브러리
import pandas as pd

# header = None, csv 파일에 속성열이 없음
df = pd.read_csv('./data/sonar3.csv', header = None)

# df.head()

# 60번째 열의 데이터를 출력
# 1 : 111, 0 : 97, Name : 60, dtype: int64
df[60].value_counts()

X = df.iloc[:, 0:60]
y = df.iloc[:, 60]

# 학습셋과 데이터셋 구분
"""
X_train: 학습용 특성 데이터(입력 데이터)
X_test: 테스트용 특성 데이터(입력 데이터)
y_train: 학습용 레이블 데이터(타겟 데이터)
y_test: 테스트용 레이블 데이터(타겟 데이터)
test_size : 테스트 데이터의 비율
random_state : 난수를 제어하는 역할로 
다른 머신 러닝에 대해 동일성을 부여하는 역할
random_state에 어떤 값이 있다면 난수 생성에 규칙을 넣어서 동일한 결과를 출력
1이라는 값을 넣어서 1 3 4 7 번째 데이터를 이용했다면, 
다시 1을 넣을 때 1 3 4 7 번째 데이터를 사용
"""
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=True)


model = Sequential()
model.add(Dense(24, input_dim = 60, activation='relu'))
model.add(Dense(10, activation = 'relu'))
# 레이어 모델이 sigmoid -- 1
model.add(Dense(1, activation = 'sigmoid'))

# 손실 함수가 이진 분류 -- 2
# 1과 2는 함께 가는 것으로 이해해보자.
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

history = model.fit(X, y, epochs = 200, batch_size = 10)

# 결과값
# Epoch 200/200
# 21/21 [==============================] - 0s 2ms/step - loss: 0.0176 - accuracy: 0.9952
# 실제로 판별해내는 확률도 정확도 99.52%일까?

score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
# 결과값
# Test accuracy: 1.0

"""
과적합은 모델이 학습 데이터셋 안에서는 일정 수준 이상의 예측 정확도를 보이지만, 
새로운 데이터에 적용하면 잘 맞지 않는다는 것을 의미함.

과적합은 층이 너무 많거나 변수가 복잡해서 발생하기도 하며, 테스트셋과 학습셋이 중복될 때 생기기도 함. 
딥러닝은 학습 단계에서 입력층, 은닉층, 출력층의 노드들에 상당히 많은 변수가 투입됨. 
딥러닝을 진행하는 동안 과적합에 빠지지 않게 주의해야 함.

과적합을 방지하기 위해서 학습을 위한 학습셋과 이를 테스트할 데이터셋을 완전히 구분하고, 
학습과 동시에 테스트를 병행하면서 진행하는 방법임.

전체 샘플을 반복할 때마다 학습셋과 데이터셋을 무작위로 선정함. 
이렇듯 무작위로 선택되었을 때, 모델이 각 반복에서 다양한 데이터를 학습하고 평가하게 하는 효과가 있음.

학습셋은 학습의 결과를 가중치에 반영하지만, 테스트셋은 과적합이 발생하는지를 확인하기 위한 지표로 
사용자를 위한 출력문으로 이해할 수 있음. 이러한 의미에서 테스트셋의 정확도는 학습의 결과에 반영되지 않음.
"""

# 새로운 데이터에 대해 높은 정확도를 안정되게 보여주는 모델을 만들기 위한 방법
# 1. 데이터를 보강 // 2. 알고리즘 최적화
# 새로운 데이터에 대해 높은 정확도를 안정되게 보여주는 모델을 만들기 위한 방법으로
# 데이터를 보강하거나 알고리즘을 최적화하는 것이 있음.
# 하지만, 데이터를 추가하는 것 자체가 어렵거나 데이터 추가만으로는 성능에 한계가 있을 수 있음. 
# 따라서, 가지고 있는 데이터를 적절히 보완해주는 방법으로 
# 사진의 경우 확대/축소한 것을 더하거나 위아래로 조금씩 이동함. 
# 테이블형 데이터의 경우, 너무 크거나 낮은 이상치가 모델에 영향을 줄 수 없도록 크기를 적절히 조절할 수 있음. 
# 시그모이드 함수를 사용해 전체를 0~1사이의 값으로 변환하는 것도 방법임.

# 알고리즘을 이용해 성능을 향상하는 방법으로는 은닉층의 개수, 노드의 수, 최적화 함수를 변경하는 것이 있음.

# 모델 저장과 재사용
# model.save(경로)
# hdf5 파일 포맷은 주로 과학 기술 데이터 작업에서 사용
model.save('./data/model/my_model.hdf5')


# 저장한 파일 불러오기
# load_model
from tensorflow.keras.models import Sequential, load_model

# 위에서 실행한 모델을 메모리에서 삭제
del model

model = load_model('./data/model/my_model.hdf5')
# X_test와 Y_test를 사용하여 학습된 딥러닝 모델을 평가 수행
# score = [loss, accuracy, precision, recall]
# 손실, 정확도, 정밀도, 리콜 값을 포함한 리스트
score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])
