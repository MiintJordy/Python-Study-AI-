from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import os

## MNIST ������ �ҷ�����
# X_train : �ȼ��� �̷���� 60,000���� ������
# y_train : �� �����Ͱ� ����Ű�� ����
# X_test : �ȼ��� �̷���� 10,000���� ������
# y_test : �� �����Ͱ� ����Ű�� ����
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## ���� ��ȯ
# 25*25�� �� �迭�� 1�������� ��ȯ
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32')/255

## ��-���ڵ�
# 0~9�� ǥ�õ� ���ڸ� 0�� 1�� ��ȯ
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

## �� ���� ����
model = Sequential()
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

## �� ���� ȯ�� ����
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## �� ����ȭ ����
# modelpath : �� ���� ���
# monitor : val_loss ��ġ�� ����͸�
# verbose = 1 : epochs�� �Ϸ�� ������ ��� ���
# save_best_only = True : ���� �� ����
modelpath = "./MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', 
                               verbose=1, save_best_only=True)

## �н� ���� ����
# monitor : val_loss ��ġ�� ����͸�
# patience : ���� 10ȸ ������ ������
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

## �� ����
# X_train, y_train�� �������� �н�
# validation_split = 0.25 : �н����� 25%
# epochs = 30 : ��ü �����͸� 30�� �ݺ�
# batch_size = 200 : ���� 200�� �н� �� ����ġ ������Ʈ
# verbose = 0 : ��� ��� X
# callbacks : ȣ���� �Լ� ���� ���� �� �� ����
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30,
                    batch_size=200, verbose=0, callbacks=[early_stopping_callback,
                                                          checkpointer])

## �׽�Ʈ ��Ȯ�� ���
# model.evaluate�� ��� [loss, accuracy]
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))

## �����°� �н����� ������ ����
# history�� ��� [loss, accuracy, val_loss, val_accuracy]
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

## �׷��� ǥ��
# x���� ���̴� len(y_loss)��ŭ�� ���̷� ����
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label="Testset_loss")
plt.plot(x_len, y_loss, marker='.', c="blue", label="Trainset_loss")

## �׷����� �׸��带 �ְ� ���̺��� ǥ��
# ���� ��ġ ����
plt.legend(loc='upper right')
# ���μ��� ���μ� �׸��� ǥ��(�� ����)
plt.grid()
# x�� ���� ����
plt.xlabel('epoch')
# y�� ���� ����
plt.ylabel('loss')
plt.show()