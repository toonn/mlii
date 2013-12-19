import numpy as np
import classifyrunner.preprocess.run as run
import classifyrunner.preprocess.dataset as dataset
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut

import cPickle as pickled
use_gurkin = True

m = dataset.load_classes('data/metadata.csv')

print 'Loading data...'
if use_gurkin:
    try:
        print 'Found gurkin'
        with open('dataset_dictionary.gurkin','r') as ddg:
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
#merge_Xs = []
#merge_Ys_trained = []
#merge_Ys_surface = []
#merge_Ls = []
#for X,Y,L in zip(Xs,Ys,Ls):
#    merge_Xs.extend(X)
#    merge_Ys_trained.extend(Y[0])
#    merge_Ys_surface.extend(Y[1])
#    merge_Ls.extend(L)
#Xs = np.asfarray([merge_Xs])
#Ys = np.asarray([[merge_Ys_trained, merge_Ys_surface]])
#Ls = [merge_Ls]

for X, Y, labels in zip(Xs, Ys, Ls):
    lolo = LeaveOneLabelOut(labels)
    dt = tree.DecisionTreeClassifier()
    scores = cross_val_score(dt, X, Y[0], cv=lolo)
    print 'Decision Tree Accuracy, trained: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
    scores = cross_val_score(dt, X, Y[1], cv=lolo)
    print 'Decision Tree Accuracy, surface: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
    sv = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(sv, X, Y[0], cv=lolo)
    print 'SVM Accuracy, trained: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
    scores = cross_val_score(sv, X, Y[1], cv=lolo)
    print 'SVM Accuracy, surface: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
    nb = naive_bayes.GaussianNB()
    scores = cross_val_score(nb, X, Y[0], cv=lolo)
    print 'Naive Bayes Accuracy, trained: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
    scores = cross_val_score(nb, X, Y[1], cv=lolo)
    print 'Naive Bayes Accuracy, surface: {mean}% (+/- {std}%)'.format(
            mean=round(100*scores.mean(),1), std=round(100*scores.std(),1))
