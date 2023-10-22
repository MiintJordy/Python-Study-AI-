"""
문장은 여러 개의 단어로 이루어져 있고, 
그 의미를 전달하려면 각 단어가 정해진 순서대로 입력되어야 함. 
즉, 여러 데이터가 순서와 관계없이 입력되던 것과는 다르게, 
먼저 입력된 데이터와 나중에 입력된 데이터 사이의 관계를 고려해야하는 문제가 생김.

이를 해결하기 위해 순환 신경망(Recurrent Neural Network, RNN) 방법이 고안됨. 
순환 신경망은 여러 개의 데이터가 순서대로 입력되었을 때, 앞서 입력받은 데이터를 잠시 기억해놓는 방법임. 
그리고 기억된 데이터의 중요성을 판단하고 별도의 가중치를 부여하고 다음 데이터로 넘어감. 
모든 입력 값에 이 작업을 순서대로 실행하므로 다음 층으로 넘어가기 전에 같은 층을 맴도는 것처럼 보임. 
이렇게 같은 층안에서 맴도는 성질 때문에 순환 신경망이라 부름.

예를 들어 인공지능 비서에게 "오늘 주가가 몇이야?" 라고 묻는다고 가정하면, 
RNN에 해당하는 순환 부분에서는 단어를 하나 처리할 때마다 기억해 다음 입력 값의 출력을 결정함.

순환되는 중에 앞서 나온 입력에 대한 결과가 뒤에 나오는 입력 값에 영향을 주는 것을 알 수 있음. 
이렇게 해야 비슷한 두 문장이 입력되었을 때, 그 차이를 구별해 출력 값에 반영할 수 있기 때문임. 
예를 들어 "어제 주가", "오늘 주가" 2개의 문장이 있다고 가정하면 한 쪽은 어제를 기준으로 한 쪽은 오늘을 기준으로 계산되야함.

RNN이 개발된 이후 결과를 개선하기 위해 LSTM(Long Short Term Memory) 방법을 함께 사용하는 기법이 현재 가장 널리 사용되고 있음. 
LSTM은 한 층 안에서 반복을 많이 해야하는 RNN의 특성 상 
일반 신경망보다 기울기 소실 문제가 더 많이 발생하고 이를 해결하기 어렵다는 단점을 보완한 방법임. 
즉, 반복되기 직전에 다음 층으로 기억된 값을 넘길지 여부를 관리하는 단계가 추가된 것임.
"""

""" LSTM을 이용한 카테고리 분류
입력된 문장 의미를 파악하는 것은 모든 단어를 종합해 하나의 카테고리로 분류하는 작업이라고 할 수 있음.

중부 지방은 대체로 맑겠으나, 남부 지방은 구름이 많겠습니다. → 날씨
올 초부터 유동성의 힘으로 주가가 일정하게 상승했습니다. → 주식
이번 선거에서는 누가 이길 것 같아? → 정치
퍼셉트론의 한계를 극복한 신경망이 다시 뜨고 있대. → 딥러닝
"""

# 로이터 뉴스 데이터 셋 불러오기
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

# 데이터 학습셋 나누기
# X_train : 각 기사에 대한 텍스트를 배열화
# y_train : 주제를 나타내는 정수 배열
# num_words : 데이터셋에서 가장 빈번하게 등장하는 1000개의 단어를 사용
# 전체 데이터에서 빈도가 가장 높은 1000개를 선정 후, 개별 샘플에 적용
# 1000개에 해당되지 않는 단어는 표시하지 않음.
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 0부터 인덱스를 세기 때문에 + 1을 하여 카테고리 개수 출력
# category = np.max(y_train) + 1
# print(category, '카테고리')
# print(len(X_train), '학습용 뉴스 기사')
# print(len(X_test), '테스트용 뉴스 기사')
# print(X_train[0])
# 46 카테고리
# 8982 학습용 뉴스 기사
# 2246 테스트용 뉴스 기사
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, ...]

# 단어 수를 100개로 맞춤
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)

# 카테고리 분류, 원-핫 인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델구조 설정
model = Sequential()
model.add(Embedding(1000, 100))
# 복잡한 상관관계로 인해 비선형 형태의 활성화 함수 사용
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

# 모델 실행 옵션
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습의 조기 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# 모델 실행
# history에는 주로 학습셋과 검증셋의 결과
history = model.fit(X_train, y_train, batch_size = 20, epochs = 200, validation_data=(X_test, y_test),
                    callbacks=[early_stopping_callback])

# evaluate를 사용했을 때는 주로 테스트셋 결과
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
# [1.2054282426834106, 0.7319679260253906]
print(model.evaluate(X_test, y_test))

# 검증셋과 학습셋의 오차를 저장
y_vloss = history.histroy['val_loss']
y_loss = history.history['loss']

# 그래프로 표현
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
