- RNN 기반의 언어 모델
- 언어 모델은 두 가지로 응용될 수 있음
    1. 실제 세상에서 어떤 임의의 문장이 존재할 확률이 어느 정도인지에 대한 스코어를 매기는 것
        - 문장이 문법적으로나 의미적으로 어느 정도 올바른지 측정할 수 있도록 해주고, 보통 자동 번역 시스템의 일부로 활용됨
    2. 새로운 문장의 생성
        - 셰익스피어의 소설에 언어 모델을 학습시키면 셰익스피어가 쓴 글과 비슷한 글을 네트워크가 자동으로 생성
    
# RNN이란?
- 기존의 신경망 구조에서는 모든 입력(과 출력)이 각각 독립적이라고 가정했지만 대부분의 경우는 이에 적합하지 않음
- 순차적인 정보를 처리
- 동일한 태스크를 한 시퀀스의 모든 요소마다 적용
- 출력 결과는 이전의 계산 결과에 영향
- RNN은 현재까지 계산된 결과에 대한 "메모리" 정보를 갖고 있다고 볼 수도 있음
<img src="http://www.wildml.com/wp-content/uploads/2015/09/rnn.jpg"/>
- x_t 는 시간 스텝(time step) t에서의 입력값임
- s_t 는 시간 스텝 t에서의 hidden state. 네트워크의 '메모리' 부분
- <img src="http://s0.wp.com/latex.php?latex=s_t%3Df%28Ux_t+%2B+Ws_%7Bt-1%7D%29&bg=ffffff&fg=000&s=0">
- 이 때 f는 보통 tanh 나 ReLU 가 사용됨
- 첫 hidden state를 계산하기 위한 s_t-1 은 0으로 설정됨
***
- s_t는 과거의 시간 스텝들에서 일어난 일들에 대한 정보를 전부 담고 있고, 출력값 o_t는 오로지 현재 시간 스텝 t의 메모리에만 의존
- 실제 구현에서는 너무 먼 과거에 일어난 일들은 잘 기억하지 못함
***
- 모든 시간 스텝에 대해 파라미터 값을 전부 공유하고 있음 (위 그림의 U, V, W)
- 이는 RNN이 각 스텝마다 입력값만 다를 뿐 거의 똑같은 계산을 하고 있다는 것을 보여줌
- 이는 학습해야 하는 파라미터 수를 많이 줄여줌
***
- 위 다이어그램에서는 매 시간 스텝마다 출력값을 내지만, 문제에 따라 달라질 수 있음
- 예를 들어, 문장에서 긍정/부정적인 감정을 추측하고 싶다면 굳이 모든 단어 위치에 대해 추측값을 내지 않고 최종 추측값 하나만 내서 판단하는 것이 더 유용할 수도 있음
- 마찬가지로, 입력값 역시 매 시간 스텝마다 꼭 다 필요한 것은 아님 RNN에서의 핵심은 시퀀스 정보에 대해 어떠한 정보를 추출해 주는 hidden state이기 때문

# RNN으로 할 수 있는 일
- 가장 많이 사용되는 RNN의 종류는 LSTM
- hidden state를 계산하는 방법이 조금 다를 뿐 나머지는 거의 같음
1. 언어 모델링과 텍스트 생성
- 주어진 문장에서 이전 단어들을 보고 다음 단어가 나올 확률을 계산해주는 모델
- 어떤 문장이 실제로 존재할 확률이 얼마나 되는지 계산
- 부수적인 효과로 생성(generative) 모델을 얻을 수 있음
- 문장의 다음 단어가 무엇이 되면 좋을지 정하면 새로운 문장을 생성할 수 있음
- 네트워크를 학습할 때에는 시간 스텝 t에서의 출력값이 실제로 다음 입력 단어가 되도록 o_t=x_{t+1}로 정해줌

2. 자동 번역 (기계 번역)
- 입력이 단어들의 시퀀스라는 점에서 언어 모델링과 비슷하지만, 출력값이 다른 언어로 되어있는 단어들의 시퀀스임
- 입력값을 전부 다 받아들인 다음에서야 네트워크가 출력값을 내보냄
- 언어마다 어순이 다르기 때문에 대상 언어의 첫 단어를 얻기 위해 전체를 봐야할 수도 있음

3. 음성 인식
- 사운드 웨이브의 음향 신호(acoustic signal)를 입력으로 받아들이고, 출력으로는 음소(phonetic segment)들의 시퀀스와 각각의 음소별 확률 분포를 추측할 수 있음

4. 이미지 캡션 생성
- CNN과 RNN을 함께 사용하여 임의의 이미지를 텍스트로 설명해주는 시스템을 만들 수 있음

# RNN 학습하기
- 학습 과정은 기존의 뉴럴넷과 크게 다르지 않음
- 다만 time step 마다 파라미터를 공유하기 때문에 기존의 backpropagation을 그대로 사용할 수는 없음
- 대신 Backpropagation Through Time (BPTT)라는 알고리즘을 사용함(추후 다룰 예정)
- vanishing/exploding gradient라는 문제 때문에 긴 시퀸스를 다루기 어려움
- LSTM 과 트릭 등을 통해 이러한 문제점 해결

# RNN - 확장된 모델들

1. Bidirectional RNN
- 시간 스텝 t에서의 출력값이 이전 시간 스텝 외에, 이후의 시간 스텝에서 들어오는 입력값에도 영향을 받을 수 있다는 아이디어에 기반
- 출력값은 앞, 뒤 두 RNN의 hidden state에 모두 의존하도록 계산됨
<img src="http://www.wildml.com/wp-content/uploads/2015/09/bidirectional-rnn-300x196.png">

2. Deep (Bidirectional) RNN
- 위의 구조에서 layer 의 개수가 늘어남
- 학습할 수 있는 capacity가 늘어나며 그만큼 필요한 학습 데이터 또한 많이 필요함
<img src="http://www.wildml.com/wp-content/uploads/2015/09/Screen-Shot-2015-09-16-at-2.21.51-PM-272x300.png">

3. LSTM
- 뉴런 대신에 메모리 셀이라고 불리는 구조 사용
- 입력값으로 이전 state h_t-1와 현재 입력값 x_t를 입력으로 받는 블랙박스 형태(???)
- 메모리 셀 내부에서는 이전 메모리 값을 그대로 남길지 지울지 정하고, 현재 state와 메모리 셀의 입력값을 토대로 현재 메모리에 저장할 값을 계산
- 긴 시퀀스를 기억하는데 매우 효과적

# RNN 구현하기

- 원문 : http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/
- 번역 : http://aikorea.org/blog/rnn-tutorial-2/

***

- 이 파트에서의 목표는 RNN을 이용하여 언어 모델(Language Model)을 만드는 것임
- m개의 단어로 이루어진 문장이 있다고 할 때, 언어 모델은 이 문장이 나타날 확률을 예측할 수 있음  
<img src="http://s0.wp.com/latex.php?zoom=1.5&latex=%5Cbegin%7Baligned%7D++P%28w_1%2C...%2Cw_m%29+%3D+%5Cprod_%7Bi%3D1%7D%5E%7Bm%7D+P%28w_i+%5Cmid+w_1%2C...%2C+w_%7Bi-1%7D%29++%5Cend%7Baligned%7D&bg=ffffff&fg=000&s=0"/>
- ex) "He went to buy some chocolate" 라는 문장이 있을 때,
- "He went to buy some" 뒤에 "chocolate" 이 나올 확률 * "He went to buy" 뒤에 "some" 이 나올 확률 * ...
- 언어 모델의 용도
    1. 점수를 매기는 매커니즘으로 활용할 수 있음
        - 기계 번역에서는 보통 여러 개의 문장을 생성하는데, 이 중 가장 확률이 높은 문장 선택
        - 음성 인식 시스템에서도 이와 비슷한 방법 적용
    2. 생성 모델(generative model)
        - 새로운 텍스트를 생성할 수 있음
        - 현재 갖고 있는 단어들의 시퀀스를 주고 결과로 얻은 단어들의 확률 분포에서 다음 단어를 샘플링하고, 문장이 완성될 때까지 계속 이 과정을 반복
- 위의 수식은 각 단어들의 확률은 이전에 나왔던 모든 단어들에 의존하고 있음
- 하지만 실제로는 계산량, 메모리 문제 등으로 long-term dependency를 효과적으로 다루지 못함
- => 긴 시퀀스를 처리하기 어려움

## 학습 데이터 전처리 과정
- 언어 모델을 학습하기 위해서는 텍스트 데이터가 필요함
- 라벨은 필요없음. 대신 텍스트 데이터를 쓰기 좋은 형태로 전처리해줘야 함
- 여기서는 reddit 의 댓글 텍스트 데이터를 이용함

1. TOKENIZE TEXT
- 단어 단위로 예측을 하기 위해서 댓글을 문장으로 *토큰화*하고, 문장을 단어 단위로 쪼개야 함
- ex) "He left!" => "He", "left", "!"
- 여기서는 NLTK의 word_tokenize와 sent_tokenize 방식 사용

2. REMOVE INFREQUENT WORDS
- 기억해야할 단어의 종류가 많아지면 그 만큼 학습 시간도 증가함
- 따라서 빈도수가 적은 단어들을 제외해야 함(학습하기도 힘들다)

3. PREPEND SPECIAL START AND END TOKENS
- 문장이 어떤 단어로 시작하고 끝나는지 알고 싶음
- 이를 위해 SENTENCE_START 토큰과 SENTENCE_END 토큰을 문장 양 끝에 추가함
- 이렇게 하면 첫 번째 토큰이 SENTENCE_START 일 때, 그 다음에 나오는 단어(문장의 첫 단어)는 무엇이 되는가

4. BUILD TRAINING DATA MATRICES(이 부분은 잘 이해가 되지 않음)
- RNN 의 input 은 string 이 아닌 vector 값임
- 그래서 words 와 indices 사이의 mapping 을 해줌: index_to_word, word_to_index
- 학습 데이터 x 를 [0, 179, 341, 416] 라고 하자(여기서 0은 SENTENCE_START 를 뜻함)
- 해당하는 라벨 y 는 [179, 341, 416, 1] 일 것임
- 우리의 목적은 다음 단어를 예측하는 것이기 때문에 y 는 x 를 한 자리 shift 하고 SENTENCE_END 토큰을 넣은 것이어야 함
- 따라서 179번 단어의 올바른 예측 값은 실제 다음 단어 값인 341이어야 함
```
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
with open('data/reddit-comments-2015-08.csv', 'rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    # Split full comments into sentences
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
```
```
Reading CSV file...
Parsed 79170 sentences.
Found 65751 unique words tokens.
Using vocabulary size 8000.
The least frequent word in our vocabulary is 'devoted' and appeared 10 times.

Example sentence: 'SENTENCE_START i joined a new league this year and they have different scoring rules than i'm used to. SENTENCE_END'

Example sentence after Pre-processing: '[u'SENTENCE_START', u'i', u'joined', u'a', u'new', u'league', u'this', u'year', u'and', u'they', u'have', u'different', u'scoring', u'rules', u'than', u'i', u"'m", u'used', u'to', u'.', u'SENTENCE_END']'
```
- 텍스트 데이터의 실제 학습 데이터
```
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
```
```
x:
SENTENCE_START what are n't you understanding about this ? !
[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]

y:
what are n't you understanding about this ? ! SENTENCE_END
[51, 27, 16, 10, 856, 53, 25, 34, 69, 1]
```