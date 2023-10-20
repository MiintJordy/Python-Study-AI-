# ���α׷� ���� ����

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# ���̺귯������ �����Ǵ� �ݹ� �� �ϳ��� ���� �н��� ���⿡ �ߴ�
from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/wine.csv', header = None)


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

# monitor = val_loss : ����͸��� ��ǥ ����(�� �ڵ忡���� ���� �������� �ս�)
# patience : ������ ��ǥ�� �輱���� �ʴ� epoch Ƚ�� ����
# ���� ��� 20���� ���������� ���� 20������ �������� ������ �н��� ���� �ߴ�
# verbose : �ߴ� ������ �����Ǿ��� �� ����� �޽����� �󼼵��� ����
# mode : ����͸��� ��ǥ�� ���� ������ ����(min �ս��� ���� ���� ��ǥ, max ��Ȯ���� ���� �������� ���� ��ǥ)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience = 20)

modelpath = "./data/model/Ch14-4-bestmodel.hdf5"

# monitor = val_loss : ���� �������� �ս��� ����͸��Ͽ� ���� ���� ���� �����ϵ��� ����
# save_best_only : ���� ���� ������ ���� �𵨸� ������ ������ ����
# �� epoch�� �ݺ��ϸ鼭 ���� ����ġ�� ���� ���� �޸𸮿� �����ϰ� �ִٰ� ���� ���� ������ ����
# ���α׷� ���� �ÿ� ���� ���Ҵ� ���� ����
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

history = model.fit(X_train, y_train, epochs = 2000, batch_size =500, validation_split = 0.25, verbose = 1,
                    callbacks = [early_stopping_callback, checkpointer])