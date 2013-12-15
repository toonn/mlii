import os
import features

class run(object):
    """Representation of one run.

    What constitutes a run:
        - name: string to identify a runner
        - number: string to identify a run <= '999'
        - two files with accelerometer measurements at hip and ankle
        name is assumed to be a directory in the directory root,
        containing two subdirectories enkel and heup which contain
        files named "DATA-###.CSV" where ### is a number e.g. 001
    """
    def __init__(self, root, name, number, nb_windows=1, window_size=256,
            window_shift=1):
        self.name = name
        self.number = number
        csv = 'DATA-{nb}.CSV'.format(nb=number.rjust(3,'0'))
        anklecsv = os.path.join(root, name, 'enkel', csv)
        hipcsv = os.path.join(root, name, 'heup', csv)
        self.ankle = features.derive(anklecsv)
        self.hip = features.derive(hipcsv)
        self.anklehip = features.derive(anklecsv, hipcsv)

