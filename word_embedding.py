""" 단어 임베딩 """

# 원-핫 인코딩을 사용할 때, 벡터의 길이가 너무 길어진다는 단점이 있음. 
# 1만 개의 단어 토큰으로 이루어진 말 뭉치를 다룰 때, 
# 이 데이터를 원-핫 인코딩으로 벡터화하면 9,999개의 0과 하나의 1로 이루어진 단어 벡터를 
# 1만개를 만들어야함. 이러한 공간적 낭비를 해결하기 위해 등장한 것이 "단어 임베딩"임

# 단어 임베딩으로 얻은 결과가 밀집된 정보를 가지고 있고 공간의 낭비가 적음. 
# 이러한 결과가 가능한 이유는 각 단어 간의 유사도를 계산하기 때문임. 
# 예를 들어 happy라는 단어는 bad보다 good에 더 가깝고, 
# cat이란 단어는 good보다는 dog에 가깝다는 것을 고려해 각 배열을 새로운 수치로 바꾸어줌.

# 단어 간 유사도를 계산하는 방법은 앞서 배운 오차 역전파가 등장함. 
# 적절한 크기로 배열을 바꾸어 주기 위해 최적의 유사도를 계산하는 학습 과정을 거침.
#  이 과정은 케라스에서 제공하는 Embedding() 함수를 사용하면 간단히 해낼 수 있음.

from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(16, 4))
# Embedding() 함수는 입력과 출력의 크기를 나타내는 두 개의 매개변수가 있어야 함.
# 앞 예제에서 Embedding(16, 4)는 입력될 총 단어 수는 16, 임베딩 후 출력되는 벡터 크기는 4로 하겠다는 의미

Embedding(16, 4, input_length = 2)
# 총 입력되는 데이터는 16개이지만, 매번 2개씩 입력

# 단어 수 16개
docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
        '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
        '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

""" 텍스트로 긍정 부정 예측 하기"""
# 텍스트 리뷰 자료를 지정
docs = ['너무 재밌네요', '최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고싶네요', '글쎄요', '별로예요', '생각보다 지루하네요',
        '연기가 어색해요', '재미없어요']

# 긍정 리뷰 1, 부정 리뷰를 0으로 지정
class = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 토큰화
token = Tokenizer()
# fit_on_texts : 주어진 텍스트 데이터에서 단어 빈도를 계산하여 딕셔너리를 생성
# (딕셔너리의 key : 단어, 값 : 해당 단어의 빈도에 따른 인덱스)
# word_index : 딕셔너리로서 단어와 해당 단어의 인덱스 정보 매핑
token.fit_on_texts(docs)
print(token.word_index)

# 결과값
# {'너무': 1, '재밌네요': 2, '최고예요': 3, '참': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, 
# '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고싶네요': 14, '글쎄요': 15, '별로예요': 16,
# '생각보다': 17, '지루하네요': 18, '연기가': 19, '어색해요': 20, '재미없어요': 21}

x = token.texts_to_sequences(docs)
print("\n리뷰 텍스트, 토큰화 결과:\n", x)

# 결과값
# 리뷰 텍스트, 토큰화 결과:
# [[1, 2], [3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21]]

"""
각 단어가 1부터 20까지 숫자로 토근화되었지만 리뷰 데이터의 토큰 수가 다름. 
딥러닝 모델에 입력하려면 학습 데이터의 길이가 동일하여야 함. 
따라서 토큰의 수를 똑같이 맞추어주는데 이 작업을 패딩(padding)이라고 함.

패딩 작업을 위해 keras는 pad_sequences() 함수를 제공하여, 
이를 사용하면 원하는 길이보다 짧은 부분은 숫자 0을 넣어서 채워주고 긴 데이터는 잘라서 같은 길이로 맞춤.
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_x = pad_sequences(x, 4)
print("\n패딩 결과:\n", padded_x)

# 패딩 결과:
# [[ 0  0  1  2]
# [ 0  0  0  3]
# [ 4  5  6  7]
# [ 0  8  9 10]
# [11 12 13 14]
# [ 0  0  0 15]
# [ 0  0  0 16]
# [ 0  0 17 18]
# [ 0  0 19 20]
#  [ 0  0  0 21]]

단어 임베딩을 포함해 딥러닝 모델을 만들고 결과를 출력하고자 함.
word_size = len(token.word_index) + 1
# word_size만큼 입력, 8칸으로 출력, 크기가 4인 배열을 사용
Embedding(word_size, 8, input_length=4)

""" 예제 코드 """
# 토큰화
from tensorflow.keras.preprocessing.text import Tokenizer
# 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 모델 Sequential
from tensorflow.keras.models import Sequential
# 레이어, 2차원->1차원 배열, 임베딩
from tensorflow.keras.layers import Dense, Flatten, Embedding
# 원-핫 인코딩
from tensorflow.keras.utils import to_categorical
# array 함수를 사용하여 리스트나 다른 시퀀스 데이터를 Numpy 배열로 변환
from numpy import array

# 텍스트 리뷰 자료를 지정
docs = ['너무 재밌네요', '최고예요', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다.',
        '한 번 더 보고싶네요', '글쎄요', '별로예요', '생각보다 지루하네요',
        '연기가 어색해요', '재미없어요']

# 긍정 리뷰는 1, 부정 리뷰는 0으로 지정
classes = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# 토큰화
token = Tokenizer()
# 딕셔너리의 key: 단어, 값: 해당 단어의 빈도에 따른 인덱스
# 가장 빈번한 단어가 index 1
token.fit_on_texts(docs)
# print(token.word_index) : 단어와 해당 단어의 인덱스 정보 매핑

x = token.texts_to_sequences(docs)
# [[1, 2], [3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14], 
# [15], [16], [17, 18], [19, 20], [21]]

# 패딩 : 길이 맞추기
padded_x = pad_sequences(x, 4)
# 패딩 결과:
# [[ 0  0  1  2]
# [ 0  0  0  3]
# [ 4  5  6  7]
# [ 0  8  9 10]
# [11 12 13 14]
# [ 0  0  0 15]
# [ 0  0  0 16]
# [ 0  0 17 18]
# [ 0  0 19 20]
#  [ 0  0  0 21]]

# 임베딩에 사용할 단어의 수 지정
word_size = len(token.word_index) + 1

# 단어 임베딩을 포함한 딥러닝 모델
model = Sequential()
# word_size만큼 입력, 8칸으로 출력, 크기가 4인 배열을 사용
model.add(Embedding(word_size, 8, input_length = 4))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)
print("\n Accuracy : %.4f" % (model.evaluate(padded_x, classes)[1]))