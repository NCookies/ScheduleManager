#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np

import csv
import itertools

import nltk


# cost 함수 구현
def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim  # 단어장의 크기
        self.hidden_dim = hidden_dim  # 레이어의 크기. 깊다는 의미가 아니라 t 가 많다는 의미인 듯
        self.bptt_truncate = bptt_truncate  # 나중에 설명

        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        """
        :param x: X_train[10] 
        :return: 
        """
        # The total number of time steps
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))  # hidden_state 저장 matrix 생성
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))

        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))  # hidden layer
            o[t] = softmax(self.V.dot(s[t]))  # 네트워크의 출력값

        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)  # 확률이 가장 가장 큰 단어 return


if __name__ == "__main__":
    vocabulary_size = 8000  # 단어장 크기
    unknown_token = "UNKNOWN_TOKEN"  # 단어장에 등록되지 않은 단어
    sentence_start_token = "SENTENCE_START"  # 문장의 처음에 삽입하기 위한 토큰
    sentence_end_token = "SENTENCE_END"  # 문장의 마지막에 삽입하기 위한 토큰

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print "Reading CSV file..."
    with open('../data/reddit-comments-2015-08.csv', 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        # 텍스트를 tokenize 하기 위하여 ntlk 의 sent_tokenize 사용 (comment -> sentence)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    # Tokenize the sentences into words
    # 텍스트를 tokenize 하기 위하여 ntlk 의 word_tokenize 방식 사용 (sentence -> token)
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    np.random.seed(10)
    model = RNNNumpy(vocabulary_size)
    o, s = model.forward_propagation(X_train[10])

    print o.shape
    print o
