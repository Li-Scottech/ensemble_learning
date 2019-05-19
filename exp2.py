import random
import pandas as pd
import numpy as np
from sklearn import tree, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

BAGGING_ITER = 2000
BAGGING_BATCH = 1500
ADA_ITER = 100

train = pd.read_csv('train.csv', sep='\t')
X = train[['reviewerID', 'asin', 'reviewText', 'overall']]
Y = train[['label']]

# word2vec功能，此处已经提前训练加载好了
# vector = word2vec(X, False, 'vec')

vec = np.load('vec.npy')  # 训练集向量
df = pd.DataFrame(vec)
X = pd.concat([X, df], axis=1)
X['reviewText'] = [len(line.strip().split()) for line in X['reviewText']]  # reviewLen

x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.25, random_state=1)


def naive(classifier):
    classifier.fit(x_train, y_train)
    print(classification_report(y_validate, classifier.predict(x_validate)))


def Bagging(classifier):
    size = BAGGING_ITER
    output = []
    predict = []
    for i in range(size):
        sample = []
        # bootstrap
        for j in range(BAGGING_BATCH):
            r = random.randint(0, len(x_train) - 1)
            sample.append(r)
        x_tmp = x_train.iloc[sample]
        y_tmp = y_train.iloc[sample]
        classifier.fit(x_tmp, y_tmp)
        output.append(classifier.predict(x_validate))
    for i in range(len(y_validate)):
        cnt = 0
        for j in range(size):
            if (output[j][i]) == 1:
                cnt += 1
        if cnt > size / 2:
            predict.append(1)
        else:
            predict.append(0)
    print(classification_report(y_validate, predict))


def AdaBoost(classifier):
    time = ADA_ITER  # 迭代次数(100比较慢)
    output = []
    predict = []
    w = np.ones(len(x_train)) / len(x_train)
    b = np.zeros(time)

    for t in range(time):
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
        y_pred = classifier.predict(x_validate)
        output.append([np.log(1/b[t]) if y == 1 else -np.log(1/b[t]) for y in y_pred])

    # 加权求和
    for i in range(len(y_validate)):
        cnt = 0
        for j in range(time):
            cnt += output[j][i]
        if cnt > 0:
            predict.append(1)
        else:
            predict.append(0)

    print(classification_report(y_validate, predict))


if __name__ == '__main__':
    # 请根据选择测试

    print('naive decision tree')
    # naive(tree.DecisionTreeClassifier(max_depth=6))

    print('Bagging + DecisionTree')
    # Bagging(tree.DecisionTreeClassifier(max_depth=6))

    print('AdaBoost + DecisionTree')
    # AdaBoost(tree.DecisionTreeClassifier(max_depth=6))

    print('naive SVM')
    # naive(svm.LinearSVC(dual=False))

    print('Bagging + SVM')
    # Bagging(svm.LinearSVC(dual=False))

    print('AdaBoost + SVM')
    # AdaBoost(svm.LinearSVC(dual=False))

    print('naive GaussianNB')
    # naive(GaussianNB())

    print('Bagging + GaussianNB')
    # Bagging(GaussianNB())

    print('AdaBoost + GaussianNB')
    # AdaBoost(GaussianNB())
