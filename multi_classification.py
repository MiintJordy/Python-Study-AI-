from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas ������ ���̺귯��
import pandas as pd

# seaborn �ð�ȭ ���̺귯��
import seaborn as sns

# matplotlib �ð�ȭ ���̺귯��
import matplotlib.pyplot as plt
df = pd.read_csv('./data/iris3.csv')

# ���� ���Է½�, ������ ���� 5�� ���
# df.tail() : ���� ���Է½�, ������ ���� 5�� ���
# df.head()

# sns.pairplot() : �������������� ��� ������ ���� ���� ������ ����� ����
# df : ������ ����� �׸� ������ ������
# hue = 'speices' : hue �Ű������� �����͸� �׷�ȭ�ϰų� ������ �����ϴµ� ���
# speices : �ش� ���� �� �� �����Ͱ� � ǰ���� ���ϴ��� ��Ÿ���� ��
# Ư�� �׷����� ����� ��, sns.pairplot(df, hue='species', vars=['sepal_length'])
#sns.pairplot(df, hue='species')
#plt.show()

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# ����Ͽ��� ��, y�� ���� ���ڰ� �ƴ� ���ڷ� ����Ǿ��־� ���������� ��ȯ�� �ʿ�
# print(X[0:5])
# print(y[0:5])

# ǰ�� ���� ���ؼ� ���� ���ָ� ��Ÿ���� �� ���� ���� ��ȯ
# (��-�� ���ڵ� ó��)
# [1, 0, 0] [0, 1, 0] [0, 0, 1]�� ���·� ���� 1, ���� Ȱ��ȭ�� �ǹ�
y = pd.get_dummies(y)
# print(y[0:5])

model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
# softmax
# sigmoid�� ���, 0���� 1���� ���� ���� ��ȯ�ϰ� 0.5���ϴ� ~, �̻��� ~
# ������, �� �ڵ��� ��� ǰ�� 3�� �߿� �����ؾ� �ϹǷ� softmax�� ���
# softmax�� 0���� 1 ������ ���� ��ȯ�ϰ� �� ���ô� ���� Ȯ���� �� ���� 1�� ���·� ��ȯ
# ���� ��� 0.2 + 0.7 + 0.1 = 1
model.add(Dense(3, activation='softmax'))
# ������ ���� ��� ����: ���̾� �̸�(Ÿ��), �� ���� ��� ���, ���̾� �Ķ���� ��
# model.summary()

# �׷��� sigmoid�� ���� �з��� �ս� �Լ��� ���� �з� binary_crossentropy�� ���
# softmax�� ���� �з��� categorical_crossentropy�� ���
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# �� ����
histroy = model.fit(X, y, epochs=50, batch_size=5)
