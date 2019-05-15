import os
import re
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emotions).replace('-', '')
    # tokenized = [w for w in text.split() if w not in stop]
    return text


filterate = re.compile('[^\w\u4e00-\u9fff]+')


def word2vec(X, init, s):
    lines = []
    for i in X['reviewText']:
        i = filterate.sub(r' ', i)
        # simplify the whitespace
        i = re.sub('\s\s+', ' ', i)
        i = i.split(' ')
        lines.append(i)
    if init is True:
        model = Word2Vec(lines)
        model.save('./model')
    model = Word2Vec.load('./model')
    print(model)
    vec = []
    print(len(lines))
    i = 0
    for line in lines:
        if i % 1000 == 0:
            print(i)
        i += 1
        v = np.zeros(100)
        cnt = 0
        for w in line:
            try:
                v += model[w]
                cnt += 1
            except:
                pass
        v /= cnt
        vec.append(v)
        # print(v)
    np.save(s, vec)


def tfidf(X):
    vector = TfidfVectorizer()
    v = vector.fit_transform(X['reviewText'])
    print(v)
    X['reviewText'] = v

