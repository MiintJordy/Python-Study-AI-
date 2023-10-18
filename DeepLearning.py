# tensorlow ��Ű�� �������� ���� package��
# keras�� ��� �������� package�� �θ���, 
# ���� package + package�� C++���� ���̺귯���� ������ �� ����
# moels, layers�� ����̶� �θ���, C++���� ���� ������ �ϱ� ���� �ҽ� ���Ͽ� ������ �� ����
# import Sequential, Dense�� ��⿡ �ִ� �Լ��� �ǹ��ϸ�,
# C++���� �ҽ� ���Ͽ� �ۼ��� �Լ��� ������ �� ����
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ���̺귯���� ���Ե� numpy�� �Լ��� ����ϰڴٴ� �ǹ���
# include<iostream> �� cout = numpy
# �ҷ��� �Լ� numpy�� np��� �̸����� ����ϰڴٴ� �ǹ���
import numpy as np

# ����꿡 �غ�� �����͸� ������
!git clone https:#github.com/taehojo/data.git

# ThoraricSurgery3.csv ������ �ҷ��ͼ� "," �����ڷ� numpy �迭�� �����
Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimter=",")
# �Ʒ� �����ͼ�: ���� �н��ϴµ� ���Ǵ� ������
# ���� �����ͼ�: ���� ������ ���ϰ� �������Ķ���͸� �����ϴµ� ���Ǵ� ������
# �׽�Ʈ �����ͼ�: ���� ���� ������ ���ϴµ� ���Ǵ� ������

# X�� ���� �Ӽ� �ε��� 0���� 15������ y�� ��ü �������� ���� ����� ����
X = Data_set[:, 0:16]
# X�� ���� �ε��� 16(���/���� ����)�� y�� ��ü �����͸� ����
y = Data_set[:, 16]

# ���� ����
# 1. Sequential : model.add�� ������ ���̾ ���������� ����.
# 2. Functional API: ���̾ ���� ������ ������ ������ �� �־� 1���� 1��, 2���� 2���� �ΰ�
#                    1������ 2���� 2���� Layer�� ������ �� ����.
# 3. ����Ŭ���� ��: ����� ���� ���� ���� ��, ������
model = Sequential()


# model.add: ���̾ �߰�
# model.add(���̾� ����(units, input_dim = , activation = ))

# ���̾��� ����
# Dense(Fully Connected) ���̾�: �� ������ ���� ���̾��� ��� ������ ����Ǵ� ���̾�
# Convolutional ���̾�: �ռ��� ������ ����Ͽ� �̹����� ���� �����Ϳ��� Ư¡�� ����
# Recurrent ���̾�: ���� �����Ϳ� ���� �𵨸��� ���Ǹ�, LSTM �� GRU�� ���� ������ ����
# Dropout ���̾�: �������� �����ϱ� ���� ���Ǿ�, �Ϻ� ������ �������� ��Ȱ��ȭ

# units: �ش� ���̾ ���� ���� Ȥ�� ������ ���� ����
# input_dim: �Է� �������� ������ �����ϴ� �Ű������� �Է� �������� Ư�� ���� ����� ������ �ǹ�

# activation: Ȱ��ȭ �Լ�
# ReLU(Rectified Linear Unit): �ַ� ���� ���̾�� ���Ǹ�, ������ ����
# Sigmoid: ���� �з� �������� ��� ���̾ ���
# Softmax: ���� Ŭ���� �з� �������� ��� ���̾ ���, Ȯ�� ������ ����
# tanh: ���� ��ȯ �Ű��(RNN)���� ����ϰ� -1�� 1������ ���� ���
model.add(Dense(30, input_dim = 16, activation='relu'))
model.add(Dense(1, activation='sigmold'))

# model.compile: �� ������
# model.compile(loss =, optimizer =, metrices =[])
# loss: �ս� �Լ�
# Binary Cross-Entropy: ���� �з� ������ ���
# Categorical Cross-Entropy: ���� Ŭ���� �з� ������ ���
# Mean Squared Error(��� ���� ����): ȸ�� ������ ���
# Hinge Loss: SVM�� ���� �з� ������ ���

# optimizer: ��Ƽ������
# Adam: ����ȭ �˰���
# Stochastic Gradient Descent(SGD): ������ ����ȭ �˰���, Ȯ���� ��� �ϰ���
# RMSprop: Adgrad�� �������� �н����� �����ϴ� ����� ����

# Metrices: ���� ������ ���ϱ� ���� ��ǥ
# ��Ȯ��(Accuracy), ���е�(Precision), ������(Recall), F1 ����(F1 Sceore)
# ȸ�� ���� ���, ��� ���� ����, ��� ���� ���� ���� ����
model.compile(loss='binary_crossentropy', optimizer='adam', metrices =['accuracy']);



# history�� ���� ����
# �н� �������� �߻��� ������ �����ϸ�, �н� �� ������ �ս�(loss) �� ���� ��ǥ(metrics)

# model.fit: ������ �� �� ����
# X: �Է� �����ͷ� ���� �н��� ������
# y: ��� �����ͷ� X�� ���� ������ ����� y�� ���� ��,
#    �ս��� ����Ͽ� ������ �˰������� ����ġ ������Ʈ
# Epochs: �����ͼ� ��ü�� �н��ϴ� Ƚ��
# Batch Size: n���� �н��� ������ ��, �� ����ġ ������Ʈ�� ����
history = model.fit(X, y, epochs = 5, batch_size = 16)