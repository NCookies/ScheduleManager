참고 </br>
*http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/* - 원문 </br>
*http://mlduck.tistory.com/6* - 번역

# DEEP LEARNING FOR CHATBOTS - Part 1

## A TAXONOMY OF MODELS

### RETRIEVAL-BASED VS. GENERATIVE MODELS

**검색 기반 모델(easier)**
- 입력과 문맥에 기반한 적절한 응답
- 대답이 선-정의(predefine) 되어 있음
- 새로운 대답을 생성하지 않음
- 문법적 실수 X
- 대화가 자연스럽지 않음

**생성 모델(harder)**
- 대답이 선-정의(predefine) 되어 있지 않음
- 밑바닥부터 새로운 응답 생성
- 일반적으로 기계 번역 기술에 기반함
- 일반적으로 훈련이 어렵고 필요한 데이터 양이 많음
- 문법 실수 가능성이 있음

### LONG VS. SHORT CONVERSATIONS

- 당연한 이야기지만 대화가 길어질수록 자동화하기 어려움
- Short-Text Conversations (easier) : 단일 입력 - 단일 응답

### OPEN DOMAIN VS. CLOSED DOMAIN

**open domain(easier)**
- 사용자가 대화를 어디서든 할 수 있음
- 다양한 토픽과 방대한 지식 필요
- ex) twitter, reddit 같은 SNS 등

**closed domain(harder)**
- 특정 문제만을 처리함
- 쉬운 대신 여러 방면으로 한계가 있음
- ex) technical support, shopping assistants

***

## COMMON CHALLENGES

대화 에이전트를 만들기 위해 해결해야할 다소 명백하지 않은 문제들로, 대부분 활발하게 연구되고 있음

### INCORPORATING CONTEXT(문맥 합치기)

- 합리적인 응답 시스템을 위해서는 linguistic context 와 physical context 를 모두 포함해야 함
- linguistic context : 긴 대화에서 사람들이 무엇을 말했는지, 어떤 정보가 교환되었는지 등
- 이에 대한 대부분의 접근법은 word-embeded 이지만 긴 문장에서는 적용이 어려움
- 날짜 / 시간, 위치 또는 사용자에 대한 정보와 같은 다른 유형의 상황 별 데이터를 통합해야 할 수도 있음

### COHERENT PERSONALITY(성향 일관성)
- 대화 에이전트는 의미적으로 동일한 질문에 대해 동일한 대답을 해야 함
- 많은 시스템은 linguistic하게 그럴듯한 응답을 생성하도록 학습하지만, 의미적으로 일관성있게 생성하도록 학습하지는 않음
- A Persona-Based Neural Conversation Model는 명시적으로 성향을 모델링하는 방향에 대한 첫걸음 

### EVALUATION OF MODELS
- 대화 에이전트를 평가하는 이상적인 방법은 임무를 달성했는지 확인하는 것
- 하지만 사람이 일일이 해야하기 때문에 레이블을 얻기 힘듦
- 특히 opend domain 처럼 잘 정의된 목적이 없을 때는 더더욱...

### INTENTION AND DIVERSITY
- 생성 모델의 공통적인 문제는 많은 입력에 잘 어울리는 일반적인 대답을 하는 경향이 있다는 것임
- ex) "That's great!", "I don't know"
- 다양성을 증진하기 위해 여러 시도
- 그러나 인간은 일반적으로 입력에 대해서 특정 응답을 하고 의도를 담음
- 특정 의도를 가지도록 훈련되지 않았으므로, 이러한 종류의 다양성이 부족

***

## HOW WELL DOES IT ACTUALLY WORK?

  검색기반 모델(retrieval based model)은 사실상 opend domain 에서의 사용이 불가능하다. 
  사람이 모든 경우에 대해 대답을 준비할 수 없기 때문이다.
  그리고 opend domain 에서의 생성 모델(generative model)은 거의 AGI(Artificial General Intelligence, 강인공지능)이다.
  모든 경우를 다루어야하기 때문이다.
  대화가 길어질수록 문맥이 중요해지고 문제도 더 어려워진다.

  다음은 최근 Andrew Ng 교수의 인터뷰이다.

>Most of the value of deep learning today is in narrow domains where you can get a lot of data. Here’s one example of something it cannot do: have a meaningful conversation. There are demos, and if you cherry-pick the conversation, it looks like it’s having a meaningful conversation, but if you actually try it yourself, it quickly goes off the rails.

</br></br>

참고 </br>
*http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/* - 원문
*http://mlduck.tistory.com/7* - 번역

# DEEP LEARNING FOR CHATBOTS - Part 2

- 이번에는 검색기반 봇을 구현할 것임
- 생성 모델이 좀 더 유연한 반응을 끌어낼 수 있지만 실용화 단계는 아님
- 수 많은 훈련 데이터가 필요하고 최적화가 어렵기 때문
- 현존하는 대부분의 챗봇은 검색기반 또는 검색기반과 생성 모델을 결합한 것임
- **그렇다면 Schedule Manager 에서는 일정 관리 대화만 검색기반으로 하고 나머지 도메인은 생성 모델로 하는건 어떨까?**

## data set

- buntu Dialog Corpus (UDC) 는 이용가능한 가장 큰 공개 대화 데이터셋 중 하나
- 훈련 데이터는 1,000,000 개의 예제와 50% 긍정 (label 1), 50% 부정 (label 0)으로 이루어져있음
- 각 예제는 문맥과, 그 시점까지의 대화, 발언utterance, 문맥에 대한 응답으로 구성
- 긍정은 실제 문맥에 대한 옳은 응답인 것이고 부정은 정답 외에 랜덤으로 뽑음
- 모델 평가 방법 : **reacll@k**
  - 모델이 10개의 가능한 응답 중 k개의 좋은 응답을 고르도록 함
  - 이 중에서 정답이 있다면 그 예제는 정답 처리됨
  - 따라서 k가 커질수록 정답률이 높아짐
  - k=10 이면 100% 의 recall 을 얻음
  - k=1 이면, 모델은 정답 응답을 고를 단 한번의 기회밖에 없음
  - 이 데이터셋에서 9 distractors는 랜덤하게 골라졌지만, 실제 세계에서는 몇 백만개의 가능한 응답이 있을 수 있고, 어느 것이 옳은지 모름
  - 이 모든 것을 평가하는 것은 비용이 너무 큼
  - 아니면 가능한 응답이 몇 백개 정도 밖에 없다면 모두 평가할 수 있음
  - Google Smart Reply 는 클러스터링 기술을 사용하여 처음부터 선택할 수있는 일련의 가능한 응답을 제시함

## BASELINES
- 어떤 종류의 성능을 원하는지 이해하기 위해 간단한 baseline 모델(자세한 설명은 주석에)
```
def evaluate_recall(y, y_test, k=1):    # recall@k 알고리즘을 구현한 함수
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:    # k 개의 prediction 중 정답(label)이 있는지 확인
            num_correct += 1    // prediction 에서 앞쪽에 있을수록 높은 점수를 얻은 것임
    return num_correct/num_examples    # 정답률을 반환함
```
- first one (index 0) is always the correct one because the utterance column comes before the distractor columns in our data.
- 이 부분은 잘 이해가 되지 않음...
```
# Random Predictor
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)
    # 10개를 중복 없이 랜덤으로 추출함
# Evaluate Random predictor
y_random = [predict_random(test_df.Context[x], test_df.iloc[x,1:].values) for x in range(len(test_df))]
y_test = np.zeros(len(y_random))
for n in [1, 2, 5, 10]:
print("Recall @ ({}, 10): {:g}".format(n, evaluate_recall(y_random, y_test, n)))
```
- original paper에서 언급한 것은 random predictor가 아니라 tf-idf 임무
- term frequency – inverse document frequency : 문서에서의 단어가 전체 문서집합에서 상대적으로 얼마나 중요한지를 측정
- 직관적으로, 문맥과 응답이 비슷한 단어를 가지고 있다면, 그 둘은 올바른 쌍일 가능성이 큼
- 적어도 random predictor 보다는 가능성이 높음
- 그렇지만 여전히 만족스러운 성능은 나오지 않음
- tf-idf는 중요한 신호가 될 수 있는 단어의 순서를 무시함
- 따라서 이를 보완할 neural network를 함께 사용

## DUAL ENCODER LSTM
- Dual Encoder LSTM network
- 이 타입의 네트워크는 이 문제에 적용할 수 있는 모델 중 하나임
- 물론 가장 좋은 것은 아님
- 기계 번역 분야에서 자주 쓰이는 seq2seq 도 이 문제에 적합함
- 여기서 Dual Encoder 를 사용하는 이유는 이 문제에 대해 성능이 잘 나온다는 논문이 있기 때문(본문의 링크 참조)
<img src="http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2016/04/Screen-Shot-2016-04-21-at-10.51.18-AM-1024x690.png" alt="error">