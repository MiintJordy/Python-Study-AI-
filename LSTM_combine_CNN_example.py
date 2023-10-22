"""
model = Sequential()
# ���� �Է� ���� 0 ~ 4999�̸�, 100���� ���
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
# 5�� ũ�⸦ ���� 64���� Ŀ���� ���ÿ� ���
# padding = 'valid' : �е��� ���� ����
# padding = 1 -> �Է� �������� �� ĭ �߰�
# padding = (1, 2) -> �Է� �������� �� ĭ, �ݴ� �������� �� ĭ
# strides : Ŀ���� �� ĭ�� �̵��� ���ΰ�
model.add(Conv1D(64, 5, padding='valid', activation='relu', strides=1))
# 4ĭ���� �ִ밪
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

# �н���, �׽�Ʈ�� ������
# ���� �󵵰� ���� 5000���� �ܾ� ���
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# �ܾ� ���� ���߱�
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# ���� ������ �����մϴ�.
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
# 5�� ũ�⸦ ���� 64���� Ŀ���� ���ÿ� ���
# padding = 'valid' : �е��� ���� ����
# padding = 1 -> �Է� �������� �� ĭ �߰�
# padding = (1, 2) -> �Է� �������� �� ĭ, �ݴ� �������� �� ĭ
# strides : Ŀ���� �� ĭ�� �̵��� ���ΰ�
model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1))
# 4ĭ���� �ִ밪
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

# ���� ���� �ɼ�
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# �н��� ���� �ߴ��� ����
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# ���� ����
history = model.fit(X_train, y_train, batch_size=40, epochs=100, validation_split=0.25, callbacks=[early_stopping_callback])

# �׽�Ʈ ��Ȯ��
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# �н��°� �׽�Ʈ���� ������ ����
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# �׷��� ����
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# �׷���
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()