import random
import pandas as pd
import numpy as np
from word2vec import word2vec, tfidf
from sklearn import tree, svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier 

train = pd.read_csv('train.csv', sep='\t')
test = pd.read_csv('test.csv', sep='\t')

# print(data.shape)

X = train[['reviewerID', 'asin', 'reviewText', 'overall']]
Y = train[['label']]
test_X = test[['reviewerID', 'asin', 'reviewText', 'overall']]

# print(test_X)
# vector = word2vec(X, False, 'vec')

vec = np.load('vec.npy')
df = pd.DataFrame(vec)

X = pd.concat([X, df], axis=1)
X['reviewText'] = [len(line.strip().split()) for line in X['reviewText']]

print(X.head())

# vector = word2vec(test_X, False, 'tvec')
vec = np.load('tvec.npy')
df = pd.DataFrame(vec)

test_X = pd.concat([test_X, df], axis=1)
test_X['reviewText'] = [len(line.strip().split()) for line in test_X['reviewText']]

print(test_X.head())

x_train, y_train = X, Y
x_test = test_X

print(Y.head())


def naive(classifier):
    print(x_train.shape)
    print(y_train.shape)
    classifier.fit(x_train, y_train)
    print(classification_report(y_validate, classifier.predict(x_validate)))


def Bagging(classifier):
    print('bagging')
    size = 500
    output = []
    predict = np.zeros((len(x_test),2))
    for i in range(size):
        sample = []
        for j in range(int(len(x_train) / 500)):
                
            r = random.randint(0, len(x_train) - 1)
            sample.append(r)
        # print(sample)
        x_tmp = x_train.iloc[sample]
        y_tmp = y_train.iloc[sample]
        classifier.fit(x_tmp, y_tmp)
        # print(classifier.predict_proba(x_test))
        output.append(classifier.predict_proba(x_test))
    for i in range(size):
        predict += output[i]
    predict /= size
    return predict
    

def AdaBoost(classifier):
    time = 100
    output = []
    predict = np.zeros((len(x_test),2))
    w = np.ones(len(x_train)) / len(x_train)
    b = np.zeros(time)
    for t in range(time):
        print(t)
        classifier.fit(x_train, y_train, sample_weight=w)
        y_pred = classifier.predict(x_train)
        # error = 1.0001 - accuracy_score(y_train, y_pred)
        error = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_train['label'].iloc[i]:
                error += w[i]
        if error > 0.5:
            print('not a good classifier')
        b[t] = error / (1 - error)
        for i in range(len(x_train)):
            if y_pred[i] == y_train['label'].iloc[i]:
                w[i] *= b[t]
        # w = [w[j] * b[t] if y_pred[j] == y_train['label'][j] else w[j] for j in range(len(x_train))]
        w = w / np.sum(w)
        y_pred = classifier.predict_proba(x_test)
        output.append(np.log(1/b[t]) * y_pred)
    print(output)
    for t in range(time):
        for i in range(len(x_test)):
            predict[i][0] += output[t][i][0]
            predict[i][1] += output[t][i][1]
    for i in range(len(x_test)):
        sum = (predict[i][0] + predict[i][1])
        predict[i][0] /= sum
        predict[i][1] /= sum
    print(predict)
    return predict

if __name__ == '__main__':

    # naive decision tree
    # naive(tree.DecisionTreeClassifier())

    # Bagging + DecisionTree
    '''p = Bagging(tree.DecisionTreeClassifier())
    p = [i[1] for i in p]
    df = pd.DataFrame({'Predicted':p})
    p = pd.concat([test['Id'], df], axis=1)
    print(p)
    p.to_csv('predict.csv', index=False)'''

    # AdaBoost + DecisionTree
    p = AdaBoost(tree.DecisionTreeClassifier(max_depth=10))
    print(p)
    p = [i[1] for i in p]
    df = pd.DataFrame({'Predicted':p})
    p = pd.concat([test['Id'], df], axis=1)
    p.to_csv('predict.csv', index=False)

    # naive SVM
    # naive(svm.SVC(kernel='rbf'))
    # naive(svm.LinearSVC(dual=False))
    # naive(BaggingClassifier(svm.LinearSVC(dual=False), n_estimators=100)
    
    # Bagging + SVM
    # Bagging(svm.LinearSVC(dual=False))

    # naive Gaussian
    # naive(GaussianNB())

    print('success')
