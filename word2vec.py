import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

filterate = re.compile('[^\w\u4e00-\u9fff]+')


# init为True时重新由X训练模型 s为文件保存路径
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
    vec = []
    for line in lines:
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
    np.save(s, vec)
