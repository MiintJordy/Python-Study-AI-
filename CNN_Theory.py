""" ������� �Ű�� """
"""
������� �Ű���� �Էµ� �̹������� �ٽ� �� �� Ư¡�� �����ϱ� ���� Ŀ��(�����̵� ����)�� �����ϴ� �����.

@ �����̵� ������?
- ��ǻ�� ���� �� ��ȣ ó���� ���� �������� ���Ǵ� �Ϲ����� �����. 
�� ����� �����͸� �м��ϰų� ó���� �� �������� ���� �κ��� ���������� �̵���Ű�鼭 �۾��� ������.

4*4�� ũ��� 0�� 1�� �̷���� �̹����� �ִٸ�,
2*2 ũ���� X0 X1�� ����ġ�� ������ Ŀ���� �����,
�� ĭ�� �̵��ϸ� ���������� ��ġ�� 9���� ����� ����

3*3���� ���� ���Ӱ� ������� ���� �������(�ռ���) ���̶�� �ϸ�, 
������� ���� ����� �Է� �����Ͱ� ���� Ư¡�� �뷫������ �����ؼ� �н��� ������ �� ����.

# keras���� ������� ���� �߰��ϴ� �Լ� : Conv2D()
# 32 : 32���� Ŀ���� ���
# kernel_size(3, 3) : Ŀ�� ũ��� 3*3
# input_shape(28, 28, 1) : 28��, 28��, 1 ���  // ������ �پ��ϸ� 3 
model.add(Conv2D(32, kernel_size(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))


������ ������ ������� ���� ���� �̹��� Ư¡�� �����Ͽ���. 
������, �� ����� ������ ũ�� �����ϸ� �̸� �ٽ� �� �� ����ؾ���. �� ������ Ǯ��(pooling) 
�Ǵ� ���� ���ø�(sub sampling)�̶�� ��.

�̷��� Ǯ�� ������� ������ ���� �ȿ��� �ִ밪�� �̾Ƴ��� 
�ƽ� Ǯ��(Max Pooling)�� ��� ���� �̾Ƴ��� ��� Ǯ��(Average Pooling)�� ����.

4*4�� ǥ���� 2*2�� ũ��� ������ 4���� ������ �� �������� ���� ū ���� �����Ͽ� ���ʿ��� ������ ���߸�.

�ƽ�Ǯ���� �Ʒ��� ���� �Լ��� ����ؼ� ������ ���� ������ �� ����.

# (2,2)�� ���� 2, ���� 2 ũ���� Ǯ��â�� ���� �ƽ� Ǯ���� �����϶�� �ǹ���.
model.add(MaxPooling2D(Pool_size(2,2)))

��尡 �������ų� ���� �������ٰ� �ؼ� �н��� ������ �������� ���� �ƴϱ� ������, 
�������� �󸶳� ȿ�������� ���� �������� �ſ� �߿��Ͽ� �̿� ���� ����� �����Ǿ� ����. 
���� ���������� ȿ���� ū ����� "��� �ƿ�(Drop Out) ���"��. 
��Ӿƿ��� �������� ��ġ�� ��� �� �Ϻθ� ���Ƿ� ���ִ� ����.
�����ϰ� ��带 ���ָ� �н� �����Ϳ� ����ġ�� ġ���ļ� �н��Ǵ� �������� ������ �� ����.

# 25�� ��带 ��
model.add(Dropout(0.25)

Dense() �Լ��� �̿��� ������� �⺻ ���� ������ ��, 
������ ���� ������� ���̳� �ƽ� Ǯ���� �־��� �̹����� 2���� �迭�� ä�� �ٷ�ٴ� ����. 
�̸� 1���� �迭�� �ٲپ� �־�� Ȱ��ȭ �Լ��� �ִ� ������ ����� �� ����. 
����, Faltten() �Լ��� ����� 2���� �迭�� 1�������� �ٲپ���

���̾ ���Ǳ� ���ؼ��� 1���� �迭�� �Ǿ�� ��.
model.add(Flatten())

"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')/255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
# input_shape : 28*28 ũ���� ��� �̹���
# Conv2D(32, kernel_size(3,3)) : 3*3 ũ���� 32���� Ŀ���� ���
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))

# 3*3 ũ���� Ŀ�� 62���� ���
model.add(Conv2D(62, (3,3), activation='relu'))

# 2*2 ũ���� Ŀ�η� �ƽ�Ǯ�� ����
model.add(MaxPooling2D(pool_size=(2,2)))

# ��� 25% ����
model.add(Dropout(0.25))

# Dense���� �����ϱ� ���� ������ �����͸� 1���� �����ͷ� ��ȯ
model.add(Flatten())

# 128�� ��� ����
model.add(Dense(128, activation='relu'))

# ��� 50% ����
model.add(Dropout(0.5))

# 10�� ��� ����
model.add(Dense(10, activation='softmax'))

# softmax�ϱ� �ս��Լ��� categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# �� ����ȭ
modelpath = "./MNIST_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose = 1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 10)

# �� ����
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0,
                    callbacks=[early_stopping_callback, checkpointer])

# �׽�Ʈ ��Ȯ�� ���
print("\n Test Accuracy : %.4f" % (model.evaluate(X_test, y_test)[1]))

# ���� �°� �н����� ���� ����
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# �׷��� ǥ��
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker=".", c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker=".", c="blue", label='Trainset_loss')

# �׷����� �׸��带 �ְ� ���̺��� ǥ��
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()