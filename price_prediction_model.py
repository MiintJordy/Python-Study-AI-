# ���̾� ����������
from tensorflow.keras.models import Sequential
# ��� ��� ����
from tensorflow.keras.layers import Dense
# ���� ���� ���̺귯��
from tensorflow.keras.callbacks import EarlyStopping
# �׽�Ʈ�� ���̺귯��
from sklearn.model_selection import train_test_split
# üũ����Ʈ ���̺귯��
from tensorflow.keras.callbacks import ModelCheckpoint

# �׷��� ���
import matplotlib.pyplot as plt
# �ð�ȭ �� ��� ����
import seaborn as sns
# ������ �迭�� ���� �Լ�
import pandas as pd
# ������ ���۰� �м�
import numpy as np
# ������ �ð�ȭ
import matplotlib.pyplot as plt

# �ش� ��ο� �ִ� ������ ������ ���
df = pd.read_csv("./data/house_train.csv")
# ī�װ��� ������ ���ڷ� ����
df = pd.get_dummies(df)

# ��� �����͸� �ش� ���� ��հ����� ����
df = df.fillna(df.mean())
# ������ ������ ������踦 ����
df_corr = df.corr()
# �� ���� ������ ū �� ������ ����
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)
# ���ĵ� ������ SalePrice�� �����ϰ� ���� 5���� �� ����
cols_train = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']
# cols_train�� ���ϴ� ����Ʈ�� �Ӽ� ���� X_train_pre�� ����
X_train_pre = df[cols_train]
# SalePrice�� ���� ����
y = df['SalePrice'].values

# ��ü�� 80%�� �н���, 20%�� �׽�Ʈ������ ����
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)

# �� ����
model = Sequential()
# X_train.shape[1] : X_train�� �ִ� �Ӽ��� ����
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary()

# �� ���� (���ο� �ս� �Լ�: ȸ�Ϳ� �ַ� ���)
model.compile(optimizer='adam', loss='mean_squared_error')

# 20�� �̻� ����� ������ ������ �ڵ����� �ߴ�
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

# �� ���� ���
modelpath = "./data/model/Ch15-house.hdf5"

# �� ���� ����(���� ���, ��Ŀ��: ���������� �ս�, ���� ��� X, ���� ���� ��)
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

# �� ����
# X_train: �Է� Ư��, y_train: ��ǥ ����
# validation_split = 0.25, �н� �������� 25%�� ���� �����ͷ� ���
# epochs = 2000(ȸ �ݺ�), batch_size=32(�̴Ϲ�ġ ������ ����ġ ������Ʈ)
# callbacks : �� �н� �ֿ� ȣ��Ǵ� �ݹ� �Լ��� ����Ʈ
# early_stopping_callback: ���� ����
# checkpointer: �� ����ġ�� �����ϴ� �ݹ����� �־��� ������ �����ϴ� ��� ����
# X_train, y_train �� ����Ͽ� �н�
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2000, batch_size=32,
                    callbacks=[early_stopping_callback, checkpointer])

# �н� ��� �ð�ȭ
# ���� �� ���� ���� ����Ʈ
real_prices = []
# ���� �� ���� ���� ����Ʈ
pred_prices = []
# Ƚ���� ���� ����Ʈ
X_num = []
#���� Ƚ���� 0
n_iter = 0

# model.predict(X_test) : X_test ���� ����Ͽ� X_test�� ���� ���� ���� ����ϴ� �κ�
# flatten() : ������ �迭�� 1�������� ��ġ�� ����, ���� ����� 1���� �迭�� ���� �� ����
Y_prediction = model.predict(X_test).flatten()

# 25ȸ �ݺ�
for i in range(25):
    # ���� ������ y_test���� �����ͼ� real ������ ����
    real = y_test[i]
    # ���� ������ Y_prediction���� �����ͼ� prediction ������ ����
    prediction = Y_prediction[i]
    # ���� ���ݰ� ���� ������ �Ҽ��� 2�ڸ� ���� ���
    print("���� ����: {:.2f}, ���󰡰�: {:.2f}".format(real, prediction))
    # ���� ������ ����Ʈ�� append
    real_prices.append(real)
    # ���� ������ ����Ʈ�� append
    pred_prices.append(prediction)
    # �ݺ� Ƚ���� ����
    n_iter = n_iter + 1
    # Ƚ���� ����Ʈ�� append
    X_num.append(n_iter)

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
# ���� �߰�
plt.legend()
plt.show()