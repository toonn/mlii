import os
from run import run
import csv
import numpy as np
from sklearn import preprocessing

runners = ['Ann', 'Annick', 'Emmy', 'Floor', 'Hanne',
            'Jolien', 'Laura', 'Mara', 'Nina', 'Sofie',
            'Tina', 'Tinne', 'Vreni', 'Yllia']
trained = ['n', 'y']
surface = ['Woodchip', 'Asphalt', 'Track']

def load_data(root, nb_windows=1, window_size=256, window_shift=1):
    """Loads a dataset from the directory root.

    Every subdirectory of root is treated as one runner.
    Every runner directory contains two directories: enkel and heup
    These directories contain files in pairs: DATA-###.CSV
    """
    runs = {}
    for runner in os.listdir(root):
        runs[runner] = {}
        for i in xrange(1,10):
            nb = str(i)
            #print runner, ' -- ', nb
            try:
                r = run(root, runner, nb, nb_windows=nb_windows,
                        window_size=window_size, window_shift=window_shift)
                runs[runner][nb] = r
                print '.'
            except IOError:
                print 'x' # Who needs a helpful warning anyway.
                #print 'Not loading: {runner}-{nb}'.format(runner=runner, nb=nb)
    return runs

def load_classes(metadata):
    runmeta = {}
    with open(metadata, 'rb') as meta:
        r = csv.reader(meta, delimiter=';', skipinitialspace=True)
        rows = [row for row in r if row != [] and row[0] != 'Naam']
        for row in rows:
            runmeta[row[0]] = {}
        for row in rows:
            runmeta[row[0]].update({row[1]: (row[2], row[3])})
    return runmeta

def load_scaled_ankle(datadictionary, metadictionary):
    ankle_data = []
    ankle_meta = []
    ankle_labels = []
    nb_windows = len(datadictionary
                        .itervalues().next()
                        .itervalues().next()
                        .ankle)
    for i in xrange(nb_windows):
        ankle_data.append([])
        ankle_meta.append([[],[]])
    for runner in datadictionary.iterkeys():
        for nb in datadictionary[runner].iterkeys():
            ankle_labels.append(runners.index(runner))
            for index, window in enumerate(datadictionary[runner][nb].ankle):
                ankle_data[index].append(window)
                ankle_meta[index][0].append(
                        trained.index(metadictionary[runner][nb][0]))
                ankle_meta[index][1].append(
                        surface.index(metadictionary[runner][nb][1]))
    ankle_data = np.asfarray([preprocessing.scale(unscaled)
                                for unscaled in np.asfarray(ankle_data)])
    ankle_meta = np.asarray(ankle_meta)
    ankle_labels = [ankle_labels for i in xrange(len(ankle_meta))]
    return ankle_data, ankle_meta, ankle_labels

def load_scaled_hip(datadictionary, metadictionary):
    hip_data = []
    hip_meta = []
    hip_labels = []
    nb_windows = len(datadictionary
                        .itervalues().next()
                        .itervalues().next()
                        .hip)
    for i in xrange(nb_windows):
        hip_data.append([])
        hip_meta.append([[],[]])
    for runner in datadictionary.iterkeys():
        for nb in datadictionary[runner].iterkeys():
            hip_labels.append(runners.index(runner))
            for index, window in enumerate(datadictionary[runner][nb].hip):
                hip_data[index].append(window)
                hip_meta[index][0].append(
                        trained.index(metadictionary[runner][nb][0]))
                hip_meta[index][1].append(
                        surface.index(metadictionary[runner][nb][1]))
    hip_data = np.asfarray([preprocessing.scale(unscaled)
                                for unscaled in np.asfarray(hip_data)])
    hip_meta = np.asarray(hip_meta)
    hip_labels = [hip_labels for i in xrange(len(hip_meta))]
    return hip_data, hip_meta, hip_labels

def load_scaled_anklehip(datadictionary, metadictionary):
    anklehip_data = []
    anklehip_meta = []
    anklehip_labels = []
    nb_windows = len(datadictionary
                        .itervalues().next()
                        .itervalues().next()
                        .anklehip)
    for i in xrange(nb_windows):
        anklehip_data.append([])
        anklehip_meta.append([[],[]])
    for runner in datadictionary.iterkeys():
        for nb in datadictionary[runner].iterkeys():
            anklehip_labels.append(runners.index(runner))
            for index, window in enumerate(datadictionary[runner][nb].anklehip):
                anklehip_data[index].append(window)
                anklehip_meta[index][0].append(
                        trained.index(metadictionary[runner][nb][0]))
                anklehip_meta[index][1].append(
                        surface.index(metadictionary[runner][nb][1]))
    anklehip_data = np.asfarray([preprocessing.scale(unscaled)
                                for unscaled in np.asfarray(anklehip_data)])
    anklehip_meta = np.asarray(anklehip_meta)
    anklehip_labels = [anklehip_labels for i in xrange(len(anklehip_meta))]
    return anklehip_data, anklehip_meta, anklehip_labels

