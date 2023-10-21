# MNIST ������ �� ����(0~9)�� �����ϴ� �̹��� �����ͼ�
# ����н��� �� ���� ���� �н� �� �׽�Ʈ�ϱ� ���� �������� �����ͼ�
from tensorflow.keras.datasets import mnist
# ��-�� ���ڵ�
from tensorflow.keras.utils import to_categorical
# ������ �ð�ȭ
import matplotlib.pyplot as plt
# ��� ���ο��� ��ũ������ ������ �� �����ϴ� �μ��� ó���ϰų� ���� ȯ�� ������ ���� �� ���
import sys

# (a, b) : ��ȣ�� �ѷ��� ǥ���� Python���� Ʃ���� ��Ÿ��
# �����͸� ȿ�������� �����ϸ鼭 �������� ���̱� ���ؼ���
# �н��� ���� �κ�: X_train, y_train
# �׽�Ʈ�� ���� �κ�: X_test, y_test
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape[0] : �Ӽ��� ����
# shape[1] : ���� ����
# print("�н��� �̹��� ��: %d" % (X_train.shape[0]))
# print("�׽�Ʈ�� �̹��� ��: %d" % (X_test.shape[0]))

# �ҷ��� �̹��� �� �� ���� �ҷ��ͼ� Ȯ���غ���
# plt.imshow(X_train[0], cmap='Greys')
# plt.show()

# �Ʒ� �ڵ�� ���ڸ� 0~255���� ����� �ű� ���� Ȯ��
# for x in X_train[0]:
#     for i in x:
#         sys.stdout.write("%-3s" % i)
#     sys.stdout.write('\n')

# �־��� ���� 28, ���� 28�� 2���� �迭�� 784���� 1���� �迭�� �ٲ�
X_train = X_train.reshape(X_train.shape[0], 784)

# keras�� �����͸� 0���� 1������ ������ ��ȯ�� �� ������ �� ������ ������ ����
# ������ ����ȭ : �������� ���� Ŭ �� ������ ������ �л��� ������ �ٲٴ� ����
X_train = X_train.astype('float64')
X_train = X_train / 255

X_test = X_test.reshape(X_test.shape[0], 784).astype('float64')/255

# y_train : array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
# print("class : %d" % (y_train[0]))

# �������� �з������� �ذ��ϱ� ���ؼ��� ��-�� ���ڵ� ����� �����ؾ���
# 0~9�� ������ ���� ���� ���� ���¿��� 0 �Ǵ� 1�θ� �̷���� ���ͷ� ���� �����ؾ���
# to_categorical(label, num) : �ش� label�� ������ num���� �ǹ�
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# array([[0., 0., 0., ..., 0., 0., 0.],
#        [1., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 1., 0.]], dtype=float32)
#y_train