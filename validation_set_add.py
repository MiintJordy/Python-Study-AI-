from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

""" �� ������Ʈ """
# �н� ���� ���� �����ϴ� �Լ�
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd

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

""" �� ������Ʈ """
# 50��° epoch�� ��������Ȯ���� 0.9346�̶�� 50-0.9346.hdf5��� �̸����� �����
modelpath = "./data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5"

# �� ����ġ�� ������ ���� ��θ� ���ϰ� ���� ��Ȳ�� ����͸�
checkpointer = ModelCheckpoint(filepath=modelpath, verbose = 1)
""" """

# test set�� �������� ���� �ڵ�
# history = model.fit(X, y, epochs = 50, batch_size = 500)
# ������ �߰� ��, validation_split�� �߰�, �н��¿����� 0.25%�� 0.8*0.25 = 0.2, 20%��
# ������ ���� �׽�Ʈ �°� ���������� ����͸��� ���
""" �� ������Ʈ
callbacks = [checkpointer]
�н� �߿� �ֱ������� �� ����ġ�� �����ϴ� �Լ��� �θ�

ModelCheckpiint : ���� ����ġ�� �ֱ������� �����ϰų� ���� ���� �����Ͽ� ����
EarlyStoppint : ������ ������ ������ �� �н��� ���� �ߴ�
ReduceLROnPlateau : �н����� �������� �����Ͽ� �н��� ����ȭ
����� ���� �ݹ�
"""
history = model.fit(X_train, y_train, epochs=50, batch_size=500,
                    validation_split=0.25, verbose = 0, callbacks=[checkpointer])

score = model.evaluate(X_test, y_test)
print('Test accuracy: ', score[1])