from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# scikit-learn ���̺귯������ �����ϴ� KFole ���(���� ���� ���)
from sklearn.model_selection import KFold

# sckikkit-learn ���̺귯������ �����ϴ� accuracy_score(���� ��)
# ���� ��� [0, 1, 1, 0, 1, 1, 0, 1]
# ���� ���̺� [0, 1, 0, 0, 1, 1, 1, 1]
# ��Ȯ�� ��� accuracy = accuracy_score(���� ���, ���� ���̺�)
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_csv('./data/sonar3.csv', header=None)

X = df.iloc[:, 0:60]
y = df.iloc[:, 60]

# ���� ����
k = 5

# �����ϱ� ���� ������ ġ��ġ�� �ʵ��� ����
# 5���� fold�� �������� ��, 5���� ���� ������ ����
kfold = KFold(n_splits=k, shuffle=True)

# ��Ȯ���� ä���� �� ����Ʈ�� �غ�
acc_score = []

# ������ ���� �����ϴ� �Լ�
def model_fn():
    model = Sequential()
    model.add(Dense(24, input_dim = 60, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    return model

# K�� ���� ������ �̿��� K���� �н��� ����
# for���� ���� K�� �ݺ�
# split()�� ���� K���� �н���, �׽�Ʈ������ �и�

# kfold.split(X)�� ȣ���� ��, Line 24�� n_splits=k�� ���ǹǷ� 5���
# �н��°� �׽�Ʈ���� 5����ϰ� 5�� �ݺ��ϰڴٴ� �ǹ�
for train_index, test_index in kfold.split(X):

    # �н��°� �׽�Ʈ�¿� ���� �ʱ�ȭ
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # ������ �� ���� model�� �ʱ�ȭ
    model = model_fn()
    # �� ������
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    # �н� ���� �� �߻��ϴ� ������ ��ųʸ� ���·� history�� ����
    # loss �ս� ��, accuracy ��Ȯ�� ��, val_loss �� �н� epochs�� ���� �սǰ�, 
    # val_accuracy �� �н� epochs���� ���� �����Ϳ� ���� ��Ȯ�� ��
    # 5��� �� �����ͼ��� �� �� 100���� �ݺ��Ͽ� 500ȸ�� ����
    # verbose = 0, �н� �������� ��� ��µ� ǥ�õ��� ����
    # verbose = 1, 1����� ������ ���� 100���� �ݺ��� �� ���� ���
    # verbose = 2, ��� �ݺ��� ��µǾ� �� 500ȸ�� ��µ�
    history = model.fit(X_train, y_train, epochs = 100, batch_size=10, verbose=0)

    # �׽��� �����͸� ����Ͽ� �н��� ���� ���ϰ� �� ��ǥ�� ��ȯ�ϴ� �޼���
    # �ַ� �սǰ� ������ ��Ʈ���� ��ȯ�Ͽ� [loss, accuracy]�� ������.
    accuracy=model.evaluate(X_test, y_test)[1]
    # 1�� �ε����� ��Ȯ���� ����Ʈ�� append
    acc_score.append(accuracy)

# ��� ��Ȯ�� = ��Ȯ���� �� / 5
avg_acc_score = sum(acc_score) / k
print('��Ȯ��:', acc_score)
print('��Ȯ�� ���:', avg_acc_score)