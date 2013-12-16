import numpy as np
import classifyrunner.preprocess.run as run
import classifyrunner.preprocess.dataset as dataset
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import LeaveOneLabelOut

runners = ['Ann', 'Annick', 'Emmy', 'Floor', 'Hanne',
            'Jolien', 'Laura', 'Mara', 'Nina', 'Sofie',
            'Tina', 'Tinne', 'Vreni', 'Yllia']
trained = ['n','y']
surface = ['Woodchip', 'Asphalt', 'Track']

m = dataset.load_classes('data/metadata.csv')

print 'Loading data...'
nb_windows = 7
d = dataset.load_data('data/Runs', nb_windows=nb_windows, window_shift=128)

print 'Scaling data...'
Xs, Ys, Ls = dataset.load_scaled_anklehip(d,m)
test1X = []
test1Y = [[],[]]
test1L = []
test2X = []
test2Y = [[],[]]
test2L = []
test3X = []
test3Y = [[],[]]
test3L = []
for nb in xrange(nb_windows):
    Ytrained = Ys[nb][0]
    Ysurface = Ys[nb][1]
    test1X.append(Xs[nb][0])
    test1Y[0].append(Ytrained[0])
    test1Y[1].append(Ysurface[0])
    test1L.append(Ls[nb][0])
    test2X.append(Xs[nb][1])
    test2Y[0].append(Ytrained[1])
    test2Y[1].append(Ysurface[1])
    test2L.append(Ls[nb][1])
    test3X.append(Xs[nb][2])
    test3Y[0].append(Ytrained[2])
    test3Y[1].append(Ysurface[2])
    test3L.append(Ls[nb][2])
Xs = Xs[:,3:]
Ys = Ys[:,:,3:]
Ls = np.asarray(Ls)
Ls = Ls[:,3:]

merge_Xs = []
merge_Ys_trained = []
merge_Ys_surface = []
merge_Ls = []
for X,Y,L in zip(Xs,Ys,Ls):
    merge_Xs.extend(X)
    merge_Ys_trained.extend(Y[0])
    merge_Ys_surface.extend(Y[1])
    merge_Ls.extend(L)
Xs = np.asfarray([merge_Xs])
Ys = np.asarray([[merge_Ys_trained, merge_Ys_surface]])
Ls = [merge_Ls]

featurenames = [loc + ' ' + ax + ' ' + fea
                for loc in ('Ankle','Hip','Anklehip')
                for ax in ('Ax','Ay','Az','Atotal')
                for fea in ['acc avg', 'acc std', 'peak avg', 'peak std',
                            'max psd freq', 'max psd value',
                            'fourier peak index', 'fourier peak value',
                            'peak to acc avg ratio', 'peak to acc std ratio',
                            'single ax to total avg ratio',
                            'single ax to total std ratio']]

with open('live_classification_experiment.txt','w') as f:
    for X, Y, labels in zip(Xs, Ys, Ls):
        dt = tree.DecisionTreeClassifier()
        sv1 = svm.SVC(kernel='linear', C=1)
        sv2 = svm.SVC(kernel='linear', C=1)
        nb1 = naive_bayes.GaussianNB()
        nb2 = naive_bayes.GaussianNB()
        dt.fit(X,Y.transpose())
        sv1.fit(X,Y[0])
        sv2.fit(X,Y[1])
        nb1.fit(X,Y[0])
        nb2.fit(X,Y[1])
        print >>f, 'Decision Tree'
        res = dt.predict(test1X)
        for index, label in enumerate(test1L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'SVM'
        res1 = sv1.predict(test1X)
        res2 = sv2.predict(test1X)
        res = zip(res1,res2)
        for index, label in enumerate(test1L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'Naive Bayes'
        res1 = nb1.predict(test1X)
        res2 = nb2.predict(test1X)
        res = zip(res1,res2)
        for index, label in enumerate(test1L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'Decision Tree'
        res = dt.predict(test2X)
        for index, label in enumerate(test2L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'SVM'
        res1 = sv1.predict(test2X)
        res2 = sv2.predict(test2X)
        res = zip(res1,res2)
        for index, label in enumerate(test2L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'Naive Bayes'
        res1 = nb1.predict(test2X)
        res2 = nb2.predict(test2X)
        res = zip(res1,res2)
        for index, label in enumerate(test2L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'Decision Tree'
        res = dt.predict(test3X)
        for index, label in enumerate(test3L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'SVM'
        res1 = sv1.predict(test3X)
        res2 = sv2.predict(test3X)
        res = zip(res1,res2)
        for index, label in enumerate(test3L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
        print >>f, 'Naive Bayes'
        res1 = nb1.predict(test3X)
        res2 = nb2.predict(test3X)
        res = zip(res1,res2)
        for index, label in enumerate(test3L):
            print >>f, runners[label], ': ', trained[int(res[index][0])], surface[int(res[index][1])]
