from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pandas ������ ���̺귯��
import pandas as pd
# matplotlib �ð�ȭ ���̺귯��
import matplotlib.pyplot as plt
# seaborn �ð�ȭ ���̺귯��
import seaborn as sns

df = pd.read_csv('./data/pima-indians-diabetes3.csv')

# csv ���Ͽ� �ִ� ������ ���� 5�� ���
df.head(5)

# �Ӽ��� Ư¡�� ��ġȭ
# 1. ����(count): �� ���� ������� ���� �׸��� ��
# 2. ���(mean): �� ���� ��հ�
# 3. ǥ������(std) : �� ���� ǥ������
# 4. �ּڰ�(min) : �� ���� �ּҰ�
# 5. �ִ밪(max) : �� ���� �ִ밪
# 6. 25%, 50% 75% ����� ��
df.describe()

# �� �׸��� ������踦 �ð�ȭ
# �� ���� ��� ���踦 �����Ͽ� ��� ����� ��ȯ(-1 ~ 1 ���� ����)
# 1(�Ϻ��� ���� ��� ����), 0(��� ����), -1(�Ϻ��� ���� ���� ��� ����)
df.corr()

# �����ͽð�ȭ���� ������ ������ �� ����� �� �ִ� �÷����� ����
# gist_heat : ���� ������ ���� ������ ������ ��ο� ������ ǥ��
# cm : �������� ���� ���� ������ �����ϴ� ����� �ǹ�
colormap = plt.cm.gist_heat


# �׸��� ũ�Ⱑ ���� 12inch * 12inch
plt.figure(figsize=(12,12))

# heatmap() : ��Ʈ�� �׷����� �׸��� �Լ��� ȣ��
# dr.corr() : df�� ��� ����� ���
# linewidths = 0.1 : ��Ʈ�� ���� ��(cell) ������ ������ ����
# vmax = 0.5 : ��� ����� ������ �����Ͽ� �ð�ȭ�� ��� ����
# cmap=colormap : �÷� �� ����
# linecolor = white : ��Ʈ�� �� ������ ���� ������ ����
# annot = True : �� ���� ���� ���� ǥ������ ���θ� ����
sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor='white', annot=True)

# ���̺귯������ �׷����� ȭ�鿡 ǥ���ϴ� �Լ�
plt.show()

# X�� ����, index 0 ~ 7�� ���ϴ� y�� ��ü 
X = df.iloc[:, 0:8]

# x�� ����, index 8�� ���ϴ� y�� ��ü
y = df.iloc[:, 8]

# ���̾ ���������� ���� �� ��ġ
model = Sequential()

# Dense : �Է°� ����� ��ü�� ��� ����
# 12, 8, 1 : �ش� ���̾� ���� ���� �Ǵ� ����� ��
# input_dim : �Է� �������� Ư�� ��
# activation : Ȱ��ȭ �Լ�
# name : ���̾� �̸� �ο�
model.add(Dense(12, input_dim = 8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid', name='Dense_3'))

# �� ���� �̸�(Ÿ��), ��� ���(batch_size, ��� ����), Param(����ġ ����)
# batch_size = none �� ��, ��ġ ũ�⸦ �������� ó���� �� ������ �ǹ�
model.summary()

# loss : �ս� �Լ�, optimizer : ��Ƽ������, metrics : ���� ��ǥ
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy'])

# model.fit() : ������ ���� �н��ϴ� �޼���
# X : �Է� �������� Ư��
# y : ��ǥ ��� ���̺�� ���� �����Ϸ��� ��� ��
# epochs : �н��� �ݺ� Ƚ��
# batch_size : ���� ����ġ�� ������Ʈ�ϱ� ���� �̴� ��ġ ũ��
history = model.fit(X, y, epochs=100, batch_size=5)