"""
RNN�� ���� ���� �Է°��� ���� ��, �̸� �ٷ� ó���ϴ� ���� �ƴ϶� ��� ������ ����. 
�Էµ� ������ ���� ������ �ִٸ� �̸� ��� �޾� �ξ�� ������ ��� ���� ���� �� ����.

�� �׸����� RNN�� ���� ���� �Է°��� ���� ��, �̸� �ٷ� ó���ϴ� ���� �ƴ϶� ��� ������ ����. 
�Էµ� ������ ���� ������ �ִٸ� �̸� ��� �޾� �ξ�� ������ ��� ���� ���� �� ����. 
������ ���� ��� ���� ��ü ������ ���� ����Ǿ� �־� �̸� ���� ����(context vector)��� �ϴµ�, 
�Է� ���� ���̰� �ʹ� ������� �տ��� ���� ������� �߰����� ��������ų� 
���� ���Ͱ� ��� ���� ����� ���ڴ��� �����ϱ� ���� ������ ����.

���ڴ��� ���ڴ� ���̿� ���� �����, ���� ���� ������ �� ���κ��� �Ի�� ���ھ���� ����. 
�� ���ھ ����Ʈ�ƽ� �Լ��� ����ؼ� ���ټ� ����ġ�� ����. 
�� ����ġ�� �̿��� �Է� �� �� � ���� ���������� �� �� ������. 
���� ��� "��Ų� �ʿ��� ���δ� ���ϼ� �̳׿�"���� ù ��° ��� �ܾ��� "��Ų�" �ڸ��� ���� ������ �ܾ�� 
"Attention is all you need!" �� 4��° �� "you"��� ���� �н��ϴ� ����. 
�̷��� ������� �� ��¸��� ��� �Է� ���� �η� Ȱ���ϰ� �ϴ� ���� ���ټ���. 
������ ���� ��� �Է��� ���ߵǴ� RNN�� ������ �غ��س�.
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

# �ִٺ� 5000�� �ܾ ���
# �н���, �׽�Ʈ�� �з�
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=5000)

# �ܾ��� ���̸� 500���� ����
X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)

# ���� ���̾ ���������� ����
model = Sequential()
# 0~4999 ũ���� ������ �Էµǰ�, 500���� ���
model.add(Embedding(5000, 500))
# ��� 50% �������� ����
model.add(Dropout(0.5))
# ����� ���� ���ο� ���� 64���� ���ÿ� �۵�
# ���� ���̾� ó���� ���� Sequences ���·� ��ȯ(������� ������ ������ ��� ����)
# �ؽ�Ʈ ������ �ܾ�, ���� �Ǵ� ������ ����, ����� ��ȣ ����, �̹��� ������ ��
# return_sequences=False : �� �Է� �������� ���� ���� ��� ���� �ʿ��� ��쿡 ���
model.add(LSTM(64, return_sequences=True))
# Attention ���
model.add(Attention())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# �� ���� �ɼ�
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# �н� ���� �ߴ�
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

# �� ����
history = model.fit(X_train, y_train, batch_size=40, epochs=100,  validation_data=(X_test, y_test), callbacks=[early_stopping_callback])

# �׽�Ʈ ��Ȯ��
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

# �н��°� �׽�Ʈ���� ������ ����
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# �׷��� ����
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# �׷��� ���
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()