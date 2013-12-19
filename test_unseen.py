#! /usr/bin/env python
import numpy as np
import classifyrunner.preprocess.dataset as dataset
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn import preprocessing

import cPickle as pickled
use_gurkin = True

trained = ['n', 'y']
surface = ['Woodchip', 'Asphalt', 'Track']

def load_scaled(datadictionary):
    anklehip_data = []
    anklehip_labels = []
    nb_windows = len(datadictionary
                        .itervalues().next()
                        .itervalues().next()
                        .anklehip)
    for i in xrange(nb_windows):
        anklehip_data.append([])
    for runner in datadictionary.iterkeys():
        for nb in datadictionary[runner].iterkeys():
            anklehip_labels.append(runner)
            for index, window in enumerate(datadictionary[runner][nb].anklehip):
                anklehip_data[index].append(window)
    anklehip_data = np.asfarray([preprocessing.scale(unscaled)
                                for unscaled in np.asfarray(anklehip_data)])
    anklehip_labels = [anklehip_labels for i in xrange(len(anklehip_data))]
    return anklehip_data, anklehip_labels

import sys
testdir = sys.argv[1]

m = dataset.load_classes('data/metadata.csv')

print 'Loading data...'
if use_gurkin:
    try:
        print 'Found gurkin'
        with open('dataset_dictionary.gurkin') as ddg:
            d = pickled.load(ddg)
    except IOError:
        print 'Could not find gurkin...'
        d = dataset.load_data('data/Runs')
        with open('dataset_dictionary.gurkin','w') as ddg:
            pickled.dump(d, ddg)
        print 'Pickled a new gurkin'
else:
    print "I don't like gurkins..."
    d = dataset.load_data('data/Runs')

print 'Scaling data...'
Xs, Ys, Ls = dataset.load_scaled_anklehip(d,m)
X = Xs[0]
Y = Ys[0]
L = Ls[0]

print 'Training classifiers...'
dt = tree.DecisionTreeClassifier()
dt.fit(X, Y.transpose())
sv1 = svm.SVC(kernel='linear', C=1)
sv2 = svm.SVC(kernel='linear', C=1)
sv1.fit(X, Y[0])
sv2.fit(X, Y[1])
nb1 = naive_bayes.GaussianNB()
nb2 = naive_bayes.GaussianNB()
nb1.fit(X, Y[0])
nb2.fit(X, Y[1])

print 'Loading test data...'
testd = dataset.load_data(testdir)
print 'Scaling test data...'
testXs, testLs = load_scaled(testd)
testX = testXs[0]
testL = testLs[0]
print 'Predicting test data classes...'
print 'Decision Tree'
res = dt.predict(testX)
for index, label in enumerate(testL):
    print label, index+1, ': ',
    print trained[int(res[index][0])],surface[int(res[index][1])]
print 'SVM'
resTrained = list(sv1.predict(testX))
resSurface = list(sv2.predict(testX))
res = zip(resTrained, resSurface)
for index, label in enumerate(testL):
    print label, index+1, ': ',
    print trained[int(res[index][0])],surface[int(res[index][1])]
print 'Naive Bayes'
resTrained = list(nb1.predict(testX))
resSurface = list(nb2.predict(testX))
res = zip(resTrained, resSurface)
for index, label in enumerate(testL):
    print label, index+1, ': ',
    print trained[int(res[index][0])],surface[int(res[index][1])]
