import random
import pandas as pd
import numpy as np
from word2vec import word2vec, tfidf
from sklearn import tree, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier 

BAGGING_ITER = 2000
BAGGING_BATCH = 1500
ADA_ITER = 100

train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')

X = train[['reviewerID', 'asin', 'reviewText', 'overall']]
Y = train[['label']]
test_X = test[['reviewerID', 'asin', 'reviewText', 'overall']]

# word2vec功能，此处已经提前训练加载好了
# vector = word2vec(X, False, 'vec')
vec = np.load('vec.npy')  # 训练集向量
df = pd.DataFrame(vec)
X = pd.concat([X, df], axis=1)
X['reviewText'] = [len(line.strip().split()) for line in X['reviewText']]  # reviewLen

# word2vec功能，此处已经提前训练加载好了
# vector = word2vec(test_X, False, 'tvec')
vec = np.load('tvec.npy')  # 测试集向量
df = pd.DataFrame(vec)
test_X = pd.concat([test_X, df], axis=1)
test_X['reviewText'] = [len(line.strip().split()) for line in test_X['reviewText']] # reviewLen

x_train, y_train = X, Y
x_test = test_X


def Bagging(classifier):
    size = BAGGING_ITER
    output = []
    predict = np.zeros((len(x_test), 2))
    for i in range(size):
        sample = []
        # bootstrap
        for j in range(BAGGING_BATCH):
            r = random.randint(0, len(x_train) - 1)
            sample.append(r)
        x_tmp = x_train.iloc[sample]
        y_tmp = y_train.iloc[sample]
        classifier.fit(x_tmp, y_tmp)
        output.append(classifier.predict(x_test))
    for i in range(size):
        for j in range(len(x_test)):
            if output[i][j] == 0:
                predict[j][0] += 1
            else:
                predict[j][1] += 1
    predict /= size
    return predict


def AdaBoost(classifier):
    time = ADA_ITER  # 迭代次数(100比较慢)
    output = []
    predict = np.zeros((len(x_test), 2))
    w = np.ones(len(x_train)) / len(x_train)
    b = np.zeros(time)
    for t in range(time):
        print(t)
        classifier.fit(x_train, y_train, sample_weight=w)
        y_pred = classifier.predict(x_train)

        # 统计error
        error = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_train['label'].iloc[i]:
                error += w[i]

        if error > 0.5:
            print('not a good classifier')
        b[t] = error / (1 - error)
        w = [w[i] * b[t] if y_pred[i] == y_train['label'].iloc[i] else w[i] for i in range(len(x_train))]
        w = w / np.sum(w)
        y_pred = classifier.predict_proba(x_test)
        output.append(np.log(1/b[t]) * y_pred)

    for t in range(time):
        for i in range(len(x_test)):
            predict[i][0] += output[t][i][0]
            predict[i][1] += output[t][i][1]
    for i in range(len(x_test)):
        sum = (predict[i][0] + predict[i][1])
        predict[i][0] /= sum
        predict[i][1] /= sum
    return predict


def calc(p):
    p = [i[1] for i in p]
    df = pd.DataFrame({'Predicted': p})
    p = pd.concat([test['Id'], df], axis=1)
    p.to_csv('predict.csv', index=False)


if __name__ == '__main__':

    # Bagging + DecisionTree
    p = Bagging(tree.DecisionTreeClassifier(max_depth=6))
    calc(p)

    # AdaBoost + DecisionTree
    # p = AdaBoost(tree.DecisionTreeClassifier(max_depth=6))
    # calc(p)

    # Bagging + SVM
    # p = Bagging(svm.LinearSVC(dual=False))
    # calc(p)

    # AdaBoost + SVM
    # p = AdaBoost(svm.LinearSVC(dual=False))
    # calc(p)
