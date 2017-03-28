참고 </br>
*http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/* - 원문 </br>
*http://mlduck.tistory.com/6* - 번역

### Description


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
