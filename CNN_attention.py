"""
RNN은 여러 개의 입력값이 있을 때, 이를 바로 처리하는 것이 아니라 잠시 가지고 있음. 
입력된 값끼리 서로 관련이 있다면 이를 모두 받아 두어야 적절한 출력 값을 만들 수 있음.

위 그림에서 RNN은 여러 개의 입력값이 있을 때, 이를 바로 처리하는 것이 아니라 잠시 가지고 있음. 
입력된 값끼리 서로 관련이 있다면 이를 모두 받아 두어야 적절한 출력 값을 만들 수 있음. 
마지막 셀에 담긴 값에 전체 문장의 뜻이 함축되어 있어 이를 문맥 벡터(context vector)라고 하는데, 
입력 값의 길이가 너무 길어지면 앞에서 받은 결과값이 중간에서 희미해지거나 
문백 벡터가 모든 값을 제대로 디코더에 전달하기 힘든 문제가 생김.

인코더와 디코더 사이에 층을 만들고, 새로 생긴 층에는 각 셀로부터 게산된 스코어들이 모임. 
이 스코어에 소프트맥스 함수를 사용해서 어텐션 가중치를 만듦. 
이 가중치를 이용해 입력 값 중 어떤 셀을 중점적으로 볼 지 결정함. 
예를 들어 "당신께 필요한 전부는 이턴센 이네요"에서 첫 번째 출력 단어인 "당신께" 자리에 가장 적절한 단어는 
"Attention is all you need!" 중 4번째 셀 "you"라는 것을 학습하는 것임. 
이러한 방식으로 매 출력마다 모든 입력 값을 두루 활용하게 하는 것이 어텐션임. 
마지막 셀에 모든 입력이 집중되던 RNN의 단점을 극복해냄.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from attention import Attention

import numpy as np
import matplotlib.pyplot as plt

# 최다빈도 5000개 단어를 사용
# 학습셋, 테스트셋 분류
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# 단어의 길이를 500으로 맞춤
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# 생성 레이어를 순차적으로 적층
model = Sequential()
# 0~4999 크기의 정수가 입력되고, 500개로 출력
model.add(Embedding(5000, 500))
# 노드 50% 무작위로 끄기
model.add(Dropout(0.5))
# 결과값 전송 여부에 대해 64개가 동시에 작동
# 다음 레이어 처리를 위해 Sequences 형태로 반환(순서대로 나열된 데이터 요소 집합)
# 텍스트 문서의 단어, 문장 또는 글자의 나열, 오디오 신호 샘플, 이미지 프레임 등
# return_sequences=False : 각 입력 시퀀스에 대한 단일 출력 값이 필요한 경우에 사용
model.add(LSTM(64, return_sequences=True))
# Attention 사용
model.add(Attention())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 모델 실행 옵션
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습 조기 중단
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# 모델 실행
history = model.fit(X_train, y_train, batch_size=40, epochs=100,  validation_data=(X_test, y_test), callbacks=[early_stopping_callback])

# 테스트 정확도
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 학습셋과 테스트셋의 오차를 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프 설정
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프 출력
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()