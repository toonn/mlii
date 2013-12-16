import numpy as np
import itertools
import classifyrunner.preprocess.run as run
import classifyrunner.preprocess.dataset as dataset
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut

m = dataset.load_classes('data/metadata.csv')

print 'Loading data...'
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

featurenames = [loc + ' ' + ax + ' ' + fea
                for loc in ('Ankle','Hip','Anklehip')
                for ax in ('Ax','Ay','Az','Atotal')
                for fea in ['acc avg', 'acc std', 'peak avg', 'peak std',
                            'max psd freq', 'max psd value',
                            'fourier peak index', 'fourier peak value',
                            'peak to acc avg ratio', 'peak to acc std ratio',
                            'single ax to total avg ratio',
                            'single ax to total std ratio']]


from random import random
with open('features_combination_experiment.txt', 'w') as f:
    for X, Y, labels in zip(Xs, Ys, Ls):
        indices = list(range(len(X[0])))
        for i in xrange(2,len(indices)):
            for ind_comb in itertools.combinations(indices, i):
                if random() < 0.000000000009:
                    print ind_comb
                    print >>f,[featurenames[j] for j in indices]
                    Xcomb = [feat[np.asarray(ind_comb)] for feat in X]
                    lolo = LeaveOneLabelOut(labels)
                    dt = tree.DecisionTreeClassifier()
                    scores = cross_val_score(dt, Xcomb, Y[0], cv=lolo)
                    print >>f,('DT  Accuracy, trained: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
                    scores = cross_val_score(dt, Xcomb, Y[1], cv=lolo)
                    print >>f,('DT  Accuracy, surface: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
                    sv = svm.SVC(kernel='rbf', C=1)
                    scores = cross_val_score(sv, Xcomb, Y[0], cv=lolo)
                    print >>f,('SVM Accuracy, trained: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
                    scores = cross_val_score(sv, Xcomb, Y[1], cv=lolo)
                    print >>f,('SVM Accuracy, surface: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
                    nb = naive_bayes.GaussianNB()
                    scores = cross_val_score(nb, Xcomb, Y[0], cv=lolo)
                    print >>f,('NB  Accuracy, trained: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
                    scores = cross_val_score(nb, Xcomb, Y[1], cv=lolo)
                    print >>f,('NB  Accuracy, surface: {mean}% (+/- {std}%)'
                            .format( mean=round(100*scores.mean(),1),
                                    std=round(100*scores.std(),1)))
