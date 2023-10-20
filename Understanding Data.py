from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/house_train.csv")
##  파일 데이터 출력하기
# df
##  데이터 유형 출력하기
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

## 결측치, 카테고리 변수 처리하기
# df.isnull().sum() : 각 열의 결측치 갯수
# sort_values(ascending=False) : 많은 순서대로 정렬
# head(20) : 상위 20개
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
# 카테고리형 변수를 0과 1로 이루어진 변수로 바꾸기
# 문자형이 숫자로 바뀜
df = pd.get_dummies(df)

# df.fillna(X) : 결측치를 모두 X(대입값)으로 대체
# df.mean() : 해당 열의 평균 값으로 대체
df = df.fillna(df.mean())

## 속성별 관련도 추출하기
# df.corr() : 데이터 프레임(df) 내의 열들 간의 상관관계를 계산하여 
#             상관관계 행렬을 생성하는 작업을 수행
df_corr = df.corr()
# df_corr.sort_values('SalePrice', ascending=False)
# df_corr 에서 SalePrice 열에 대한 상관관계를 기준으로 내림차순으로 정렬
df_corr_sort = df_corr.sort_values('SalePrice', ascending=False)
# SalePrice 열의 상관관계 상위 10개 값을 반환
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

# 리스트에 해당되는 속성으로 열을 지정
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF']

# pairplot : 데이터프레임의 열들 간의 관계를 시각화, 산점도 행렬을 그림
# df[cols] : df에서 cols 리스트에 있는 열들만 선택
sns.pairplot(df[cols])
plt.show