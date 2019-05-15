import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from word2vec import word2vec, tfidf
from sklearn import tree, svm, preprocessing
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
# from sklearn.ensemble import BaggingClassifier

train = pd.read_csv('train.csv', sep='\t')

# print(data.shape)

X = train[['reviewerID', 'asin', 'reviewText', 'overall']]
Y = train[['label']]

# tfidf(X)
# vector = word2vec(X, False, 'vec')

# print(X['reviewText'])
vec = np.load('vec.npy')
df = pd.DataFrame(vec)

X = pd.concat([X, df], axis=1)
X['reviewText'] = [len(line.strip().split()) for line in X['reviewText']]

# print(X)
print(X.head())
# print(Y.head())

x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size=0.25, random_state=1)


def naive(classifier):
    train_sizes,train_score,test_score = learning_curve(classifier,x_train,y_train,train_sizes=[0.1,0.2,0.4,0.6,0.8,1],cv=10,scoring='accuracy')
    train_error = 1- np.mean(train_score,axis=1)
    test_error = 1- np.mean(test_score,axis=1)
    plt.plot(train_sizes,train_error,'o-',color = 'r',label = 'training')
    plt.plot(train_sizes,test_error,'o-',color = 'g',label = 'testing')
    plt.legend(loc='best')
    plt.xlabel('traing examples')
    plt.ylabel('error')
    plt.savefig('naive.png')
    print(x_train.shape)
    print(y_train.shape)
    classifier.fit(x_train, y_train)
    # print(classification_report(y_train, classifier.predict(x_train)))
    print(classification_report(y_validate, classifier.predict(x_validate)))


def Bagging(classifier):
    print('bagging')
    size = 500
    output = []
    predict = []
    for i in range(size):
        sample = []
        for j in range(int(len(x_train) / 50)):
            r = random.randint(0, len(x_train) - 1)
            sample.append(r)
        # print(sample)
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
    time = 100
    output = []
    predict = []
    w = np.ones(len(x_train)) / len(x_train)
    b = np.zeros(time)

    plt_t = []
    for t in range(time):
        plt_t.append(t)
    plt_e = []

    for t in range(time):
        print(t)
        classifier.fit(x_train, y_train, sample_weight=w)
        y_pred = classifier.predict(x_train)
        # error = 1.0001 - accuracy_score(y_train, y_pred)
        # error = 1.0001 - accuracy_score(y_train, y_pred)
        
        error = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_train['label'].iloc[i]:
                error += w[i]
        
        plt_e.append(error)
        
        if error > 0.5:
            print('not a good classifier')
        b[t] = error / (1 - error)
        '''for i in range(len(x_train)):
            if y_pred[i] == y_train['label'].iloc[i]:
                w[i] *= b[t]'''
        w = [w[i] * b[t] if y_pred[i] == y_train['label'].iloc[i] else w[i] for i in range(len(x_train))]
        w = w / np.sum(w)
        y_pred = classifier.predict(x_validate)
        output.append([np.log(1/b[t]) if y == 1 else -np.log(1/b[t]) for y in y_pred])

    for i in range(len(y_validate)):
        cnt = 0
        for j in range(time):
            cnt += output[j][i]
        if cnt > 0:
            predict.append(1)
        else:
            predict.append(0)

    plt.plot(plt_t, plt_e)
    plt.xlabel('time')
    plt.ylabel('error')
    plt.savefig('figure.png')

    print(classification_report(y_validate, predict))


if __name__ == '__main__':

    print('naive decision tree')
    # naive(tree.DecisionTreeClassifier(max_depth=10))

    print('Bagging + DecisionTree')
    # Bagging(tree.DecisionTreeClassifier())

    # AdaBoost + DecisionTree
    AdaBoost(tree.DecisionTreeClassifier(max_depth=10))

    # naive SVM
    # naive(svm.LinearSVC(dual=False))
    # naive(BaggingClassifier(svm.LinearSVC(dual=False), n_estimators=100)

    # Bagging + SVM
    # Bagging(svm.LinearSVC(dual=False))

    # AdaBoost + SVM
    # AdaBoost(svm.LinearSVC(dual=False))

    # naive GaussianNB
    # naive(GaussianNB())

    print('naive BernoulliNB')
    # naive(BernoulliNB())

    print('Bagging+BernoulliNB')
    # Bagging(BernoulliNB())

    print('success')
