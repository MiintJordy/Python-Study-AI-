"""
텍스트를 잘게 나누는 것을 시작으로 텍스트가 준비되면 단어별, 문장별, 행태소 별로 나눌 수 있음. 
이렇듯 작게 나누어진 하나의 단위를 토큰(token)이라 함.
입력된 텍스트를 잘게 나누는 과정을 토큰화(tokenization)라고 함.
"""

# 케라스가 제공하는 text 모듈의 text_to_word_sequence() 함수를 사용하면 
# 문장을 단어 단위로 쉽게 나눌 수 있음.
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# 전처리할 텍스트를 정함.
text = '해보지 않으면 해낼 수 없다'

# 해당 텍스트를 토큰화
result = text_to_word_sequence(text)
print("\n원문:\n", text)
print("\n토큰화:\n", result)

"""
원문:
 해보지 않으면 해낼 수 없다

토큰화:
 ['해보지', '않으면', '해낼', '수', '없다']
"""

""" 빈도수 확인하기 """
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다.',
        '텍스트의 단어로 토근화해야 딥러닝에서 인식됩니다.',
        '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.']

# 토큰화 함수 지정
token = Tokenizer()
# 토큰화 함수에 문장 지정
token.fit_on_texts(docs)

# 단어의 빈도수를 계산한 결과 출력
print("\n 단어 카운트:\n", token.word_counts)
# 문장 카운트
print("\n문장 카운트:", token.document_count)
# 각 단어들이 문장에 몇 개 나오는지 세기
print("\n 각 단어가 몇 개의 문장에 포함되어 있는가:\n", token.word_docs)
# 각 단어에 매겨진 인덱스 값
print("\n각 단어에 매겨진 인덱스 값:\n", token.word_index)

# 단어 카운트:
# OrderedDict([('먼저', 1), ('텍스트의', 2), ('각', 1), ('단어를', 1), 
# ('나누어', 1), ('토큰화합니다', 1), ('단어로', 1), ('토근화해야', 1), 
# ('딥러닝에서', 2), ('인식됩니다', 1), ('토큰화한', 1), ('결과는', 1), 
# ('사용할', 1), ('수', 1), ('있습니다', 1)])

# 문장 카운트: 3

# 각 단어가 몇 개의 문장에 포함되어 있는가:
# defaultdict(<class 'int'>, {'먼저': 1, '텍스트의': 2, '토큰화합니다': 1, 
# '나누어': 1, '각': 1, '단어를': 1, '인식됩니다': 1, '딥러닝에서': 2, '토근화해야': 1, 
# '단어로': 1, '수': 1, '결과는': 1, '토큰화한': 1, '사용할': 1, '있습니다': 1})

# 각 단어에 매겨진 인덱스 값:
# {'텍스트의': 1, '딥러닝에서': 2, '먼저': 3, '각': 4, '단어를': 5, '나누어': 6, 
# '토큰화합니다': 7, '단어로': 8, '토근화해야': 9, '인식됩니다': 10, '토큰화한': 11, 
# '결과는': 12, '사용할': 13, '수': 14, '있습니다': 15

""" 단어의 원-핫 인코딩 """
# [ 0 0 0 0 0 0 0 ]
# (0인덱스) (오랫동안) (꿈꾸는) (이는) (그) (꿈을) (닮아간다)

# 각 단어가 배열 내에서 해당하는 위치를 1로 바꾸어서 벡터화할 수 있음.
# 오랫동안 = [0 1 0 0 0 0 0]
# 꿈꾸는 = [ 0 0 1 0 0 0 0 ]
# 이는 = [ 0 0 0 1 0 0 0 ]
# 그 = [ 0 0 0 0 1 0 0 ]
# 꿈을 = [ 0 0 0 0 0 1 0 ]
# 닮아간다 = [ 0 0 0 0 0 0 1 ]

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer

text = "오랫동안 꿈꾸는 이는 그 꿈을 닮아간다"

token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)

# 결과값
# {'오랫동안': 1, '꿈꾸는': 2, '이는': 3, '그': 4, '꿈을': 5, '닮아간다': 6}

x = token.texts_to_sequences([text])
print(x)
# 결과값
# [[1, 2, 3, 4, 5, 6]]

# 앞 인덱스를 0으로 만들고 비워두는 이유
# 1. 관행적인 이유: 자연어 처리에서 많은 라이브러리 및 모델은 인덱스 1부터 사용함
# 2. 원-핫 인코딩 : 0부터 시작하는 인덱스를 사용하면 더욱 효과적임
word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes=word_size)
# 단어의 벡터화가 이루어짐
# 결과값
"""
[[[0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0.]
  [0. 0. 0. 0. 0. 0. 1.]]]
"""