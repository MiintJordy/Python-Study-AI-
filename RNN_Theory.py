"""
������ ���� ���� �ܾ�� �̷���� �ְ�, 
�� �ǹ̸� �����Ϸ��� �� �ܾ ������ ������� �ԷµǾ�� ��. 
��, ���� �����Ͱ� ������ ������� �ԷµǴ� �Ͱ��� �ٸ���, 
���� �Էµ� �����Ϳ� ���߿� �Էµ� ������ ������ ���踦 ����ؾ��ϴ� ������ ����.

�̸� �ذ��ϱ� ���� ��ȯ �Ű��(Recurrent Neural Network, RNN) ����� ��ȵ�. 
��ȯ �Ű���� ���� ���� �����Ͱ� ������� �ԷµǾ��� ��, �ռ� �Է¹��� �����͸� ��� ����س��� �����. 
�׸��� ���� �������� �߿伺�� �Ǵ��ϰ� ������ ����ġ�� �ο��ϰ� ���� �����ͷ� �Ѿ. 
��� �Է� ���� �� �۾��� ������� �����ϹǷ� ���� ������ �Ѿ�� ���� ���� ���� �ɵ��� ��ó�� ����. 
�̷��� ���� ���ȿ��� �ɵ��� ���� ������ ��ȯ �Ű���̶� �θ�.

���� ��� �ΰ����� �񼭿��� "���� �ְ��� ���̾�?" ��� ���´ٰ� �����ϸ�, 
RNN�� �ش��ϴ� ��ȯ �κп����� �ܾ �ϳ� ó���� ������ ����� ���� �Է� ���� ����� ������.

��ȯ�Ǵ� �߿� �ռ� ���� �Է¿� ���� ����� �ڿ� ������ �Է� ���� ������ �ִ� ���� �� �� ����. 
�̷��� �ؾ� ����� �� ������ �ԷµǾ��� ��, �� ���̸� ������ ��� ���� �ݿ��� �� �ֱ� ������. 
���� ��� "���� �ְ�", "���� �ְ�" 2���� ������ �ִٰ� �����ϸ� �� ���� ������ �������� �� ���� ������ �������� ���Ǿ���.

RNN�� ���ߵ� ���� ����� �����ϱ� ���� LSTM(Long Short Term Memory) ����� �Բ� ����ϴ� ����� ���� ���� �θ� ���ǰ� ����. 
LSTM�� �� �� �ȿ��� �ݺ��� ���� �ؾ��ϴ� RNN�� Ư�� �� 
�Ϲ� �Ű������ ���� �ҽ� ������ �� ���� �߻��ϰ� �̸� �ذ��ϱ� ��ƴٴ� ������ ������ �����. 
��, �ݺ��Ǳ� ������ ���� ������ ���� ���� �ѱ��� ���θ� �����ϴ� �ܰ谡 �߰��� ����.
"""

""" LSTM�� �̿��� ī�װ� �з�
�Էµ� ���� �ǹ̸� �ľ��ϴ� ���� ��� �ܾ ������ �ϳ��� ī�װ��� �з��ϴ� �۾��̶�� �� �� ����.

�ߺ� ������ ��ü�� ��������, ���� ������ ������ ���ڽ��ϴ�. �� ����
�� �ʺ��� �������� ������ �ְ��� �����ϰ� ����߽��ϴ�. �� �ֽ�
�̹� ���ſ����� ���� �̱� �� ����? �� ��ġ
�ۼ�Ʈ���� �Ѱ踦 �غ��� �Ű���� �ٽ� �߰� �ִ�. �� ������
"""

# ������ ���� ������ �� �ҷ�����
from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

# ������ �н��� ������
# X_train : �� ��翡 ���� �ؽ�Ʈ�� �迭ȭ
# y_train : ������ ��Ÿ���� ���� �迭
# num_words : �����ͼ¿��� ���� ����ϰ� �����ϴ� 1000���� �ܾ ���
# ��ü �����Ϳ��� �󵵰� ���� ���� 1000���� ���� ��, ���� ���ÿ� ����
# 1000���� �ش���� �ʴ� �ܾ�� ǥ������ ����.
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)

# 0���� �ε����� ���� ������ + 1�� �Ͽ� ī�װ� ���� ���
# category = np.max(y_train) + 1
# print(category, 'ī�װ�')
# print(len(X_train), '�н��� ���� ���')
# print(len(X_test), '�׽�Ʈ�� ���� ���')
# print(X_train[0])
# 46 ī�װ�
# 8982 �н��� ���� ���
# 2246 �׽�Ʈ�� ���� ���
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, ...]

# �ܾ� ���� 100���� ����
X_train = sequence.pad_sequences(X_train, maxlen=100)
X_test = sequence.pad_sequences(X_test, maxlen=100)

# ī�װ� �з�, ��-�� ���ڵ�
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# �𵨱��� ����
model = Sequential()
model.add(Embedding(1000, 100))
# ������ �������� ���� ���� ������ Ȱ��ȭ �Լ� ���
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))

# �� ���� �ɼ�
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# �н��� ���� �ߴ� ����
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# �� ����
# history���� �ַ� �н��°� �������� ���
history = model.fit(X_train, y_train, batch_size = 20, epochs = 200, validation_data=(X_test, y_test),
                    callbacks=[early_stopping_callback])

# evaluate�� ������� ���� �ַ� �׽�Ʈ�� ���
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
# [1.2054282426834106, 0.7319679260253906]
print(model.evaluate(X_test, y_test))

# �����°� �н����� ������ ����
y_vloss = history.histroy['val_loss']
y_loss = history.history['loss']

# �׷����� ǥ��
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')

# �׷����� �׸��带 �ְ� ���̺��� ǥ��
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
