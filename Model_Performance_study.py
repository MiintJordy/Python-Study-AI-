from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
sklearn 라이브러리에서 제공되는 함수로 학습용과 테스트용을 나눔
이 함수는 데이터셋을 무작위로 섞고 지정된 비율에 따라 
학습 데이터와 테스트 데이터로 나눔
"""
from sklearn.model_selection import train_test_split

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
"""
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