"""
model = Sequential()
# 숫자 입력 범위 0 ~ 4999이며, 100개로 출력
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
# 5의 크기를 가진 64개의 커널을 동시에 사용
# padding = 'valid' : 패딩이 없이 적용
# padding = 1 -> 입력 방향으로 한 칸 추가
# padding = (1, 2) -> 입력 방향으로 한 칸, 반대 방향으로 두 칸
# strides : 커널이 몇 칸씩 이동할 것인가
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
# 4칸에서 최대값
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, LSTM, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

# 학습셋, 테스트셋 나누기
# 가장 빈도가 높은 5000개의 단어 사용
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# 단어 길이 맞추기
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# 모델의 구조를 설정합니다.
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
# 5의 크기를 가진 64개의 커널을 동시에 사용
# padding = 'valid' : 패딩이 없이 적용
# padding = 1 -> 입력 방향으로 한 칸 추가
# padding = (1, 2) -> 입력 방향으로 한 칸, 반대 방향으로 두 칸
# strides : 커널이 몇 칸씩 이동할 것인가
model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1))
# 4칸에서 최대값
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

# 모델의 실행 옵션
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습의 조기 중단을 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# 모델을 실행
history = model.fit(X_train, y_train, batch_size=40, epochs=100, validation_split=0.25, callbacks=[early_stopping_callback])

# 테스트 정확도
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 학습셋과 테스트셋의 오차를 저장
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# 그래프 설정
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# 그래프
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()