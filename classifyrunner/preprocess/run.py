import features

class run(Object):
    """Representation of one run.

    What constitutes a run:
        - name: to identify a runner
        - series of features for measurements at hip and ankle
    """
    def __init__(self, name, anklecsv, hipcsv):
        self.name = name
        self.anklefeatures = features.derive(anklecsv)
        self.hipfeatures = features.derive(hipcsv)

