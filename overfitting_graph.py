# �� �������� �׷����� Ȯ��

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

import pandas as pd
""" �׷��� ���� �Լ� �߰� """
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/wine.csv', header = None)

# ������ ����
# df

# X�� ������ �Ӽ�, y�� ������ �з�
X = df.iloc[:, 0:12]
y = df.iloc[:, 12]

# �н��°� �׽�Ʈ������ �з�
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer ='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=2000, batch_size=500,
                    validation_split=0.25)

# history�� ����� �н� ����� Ȯ��
hist_df = pd.DataFrame(history.history)
# loss accuracy val_loss val_accuracy�� �Ӽ����� epochs���� �н� ����� ����
hist_df

# val_loss�� y_vloss�� ����
y_vloss = hist_df['val_loss']

# loss�� y_loss�� ����
y_loss = hist_df['loss']

# x�� ������ ����� �迭�� ����
x_len = np.arange(len(y_loss))
# ���� �������� ��ġ : (x_len, y_vloss)
# ���� �������� ǥ�� : "o"(��), c (����), markersize(ũ��), label(����)
# s(�簢��), ^(�ﰢ��), d(���̾Ƹ��), x(����), *(��ǥ)
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Testset_loss')
# �Ʒ� �������� ��ġ : (x_len, y_loss)
# �Ʒ� �������� ǥ�� : "o"(��), c (����), markersize(ũ��), label(����)
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trainset_loss')

# ������ ��ġ�� ������ ���
# upper left, lower right, lower left
plt.legend(loc='upper right')
# x�� ����
plt.xlabel('epoch')
# y�� ����
plt.ylabel('loss')
plt.show()