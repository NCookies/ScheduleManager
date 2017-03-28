***참고:`https://www.lucypark.kr/slides/2015-pyconkr/#1`***

# 텍스트를 표현하는 방법

## 기계는 텍스트를 이해하지 못함


### 단어를 표현하는 방법

1. 이진(bianry) 표현법
    - w_i 은 i 번 단어에 대한 벡터 표현
    - one-hot vector
    - 단어 간 유사도 정의 불가능
2. BOW(bag of words)
    - 단어가 문서에 존재/존재하지 않음 -> term existance
    - 단어가 문서에 n번 존재함 -> term frequency(TF)
    - 단어에 중요도를 할당하고 문서에 등장한 단어 중요도의 가중합
    - 차원이 너무 커서 문서 간 유사도 지표의 효과가 떨어짐
    - http://darkpgmr.tistory.com/125 참고
3. WordNet
    - 단어들의 상위어(hypernym) 관계 등을 정의하는 방향성 비순환 그래프
    - 모든 용어를 포함하지 못함(전문 도메인 용어 등)
    - 신조어를 추가하기 위해서는 수동으로 유지보수 해야함

위의 방법들로는 단어를 나타내기 힘듬 </br>
=> 문맥(context)을 파악해야함

***

## 문맥을 파악하는 방법

- 단어의 의미는 해당 문맥이 담고 있음
- 문맥(context) := 정해진 구간(window) 또는 문장/문서 내의 단어들

### Co-occurrence(공기, 共起)
- 단어와 단어가 하나의 문서나 문장에서 함께 쓰임
- 정의하는 두 가지 방법
    - Term-document matrix : 한 문서에 같이 등장하면 비슷한 단어
    - Term-term matrix : 단어가 문맥 내에 같이 존재하면 비슷한 단어
- 값들이 너무 편향(skewed)되어 있음(빈도 높은 단어와 낮은 단어의 격차가 큼)
- 정보성 낮은 단어 때문에 구별하기 쉽지 않음(discriminative)

### Neural embeddings
- 문맥에 있는 단어를 예측
- 언어 모델(language model) 활용
- ["나는", "파이썬", "이", "좋다"] 다음에 뭐가 나올까? ("!", ".". "?")
- 이것을 학습할 때 neural net 사용
- word2vec 이 여기에 포함
- 문서에서의 neural embedding
    - doc2vec(paragraph vector)
    - 문서(또는 문단) 벡터를 마치 단어인 양 학습
    - 차원이 |V| 에 비해 훨씬 적어짐
    - Term frequency : 문서 벡터의 크키가 단어의 수 |V|와 같음
    - doc2vec : 문서 벡터의 크기가 단어의 수 |V| 보다 작음