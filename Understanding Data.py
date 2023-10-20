from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/house_train.csv")
##  ���� ������ ����ϱ�
# df
##  ������ ���� ����ϱ�
# df.dtypes
"""
Id                 int64
MSSubClass         int64
MSZoning          object
LotFrontage      float64
LotArea            int64
                  ...   
MoSold             int64
YrSold             int64
SaleType          object
SaleCondition     object
SalePrice          int64
Length: 81, dtype: object
"""

## ����ġ, ī�װ� ���� ó���ϱ�
# df.isnull().sum() : �� ���� ����ġ ����
# sort_values(ascending=False) : ���� ������� ����
# head(20) : ���� 20��
# df.isnull().sum().sort_values(ascending=False).head(20)
"""
PoolQC          1453
MiscFeature     1406
Alley           1369
Fence           1179
FireplaceQu      690
LotFrontage      259
GarageYrBlt       81
GarageCond        81
GarageType        81
GarageFinish      81
GarageQual        81
BsmtFinType2      38
BsmtExposure      38
BsmtQual          37
BsmtCond          37
BsmtFinType1      37
MasVnrArea         8
MasVnrType         8
Electrical         1
Id                 0
dtype: int64
"""
# ī�װ��� ������ 0�� 1�� �̷���� ������ �ٲٱ�
# �������� ���ڷ� �ٲ�
df = pd.get_dummies(df)

# df.fillna(X) : ����ġ�� ��� X(���԰�)���� ��ü
# df.mean() : �ش� ���� ��� ������ ��ü
df = df.fillna(df.mean())

## �Ӽ��� ���õ� �����ϱ�
# df.corr() : ������ ������(df) ���� ���� ���� ������踦 ����Ͽ� 
#             ������� ����� �����ϴ� �۾��� ����
df_corr = df.corr()
# df_corr.sort_values('SalePrice', ascending=False)
# df_corr ���� SalePrice ���� ���� ������踦 �������� ������������ ����
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)
# SalePrice ���� ������� ���� 10�� ���� ��ȯ
df_corr_sort['SalePrice'].head(10)

"""
SalePrice       1.000000
OverallQual     0.790982
GrLivArea       0.708624
GarageCars      0.640409
GarageArea      0.623431
TotalBsmtSF     0.613581
1stFlrSF        0.605852
FullBath        0.560664
TotRmsAbvGrd    0.533723
YearBuilt       0.522897
Name: SalePrice, dtype: float64
"""

# ����Ʈ�� �ش�Ǵ� �Ӽ����� ���� ����
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']

# pairplot : �������������� ���� ���� ���踦 �ð�ȭ, ������ ����� �׸�
# df[cols] : df���� cols ����Ʈ�� �ִ� ���鸸 ����
sns.pairplot(df[cols])
plt.show