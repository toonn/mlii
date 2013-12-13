import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import classifyrunner.accproc as ap
import operator

def _acceleration_avg_std(A):
    """Calculate average and standard deviation for acceleration meassurements

    A -- data.Ax (e.g.)

    source: HowToExtractPeaks.html
    """
    return np.mean(A.values), np.std(A.values)

def _peak_avg_std(ym):
    """Calculate the average and standard deviation for the peak accelerations

    ym -- series of magnitudes from acceleration measurements
        Ax_max = accproc.detectPeaksGCDC()
        Ax_xm,Ax_ym = zip(*Ax_max)

    source: HowToExtractPeaks.html
    """
    return np.mean(ym), np.std(ym)

def _getStepTimes(peaks):
    """Calculate the time between every two peaks.

    Keyword arguments:
    peaks -- List of times of peaks
    
    source: HowToExtractPeaks.html
    """
    steptimes = []
    for i in xrange(len(peaks)-1): # xrange is more efficient than range
        steptimes.append(peaks[i+1][0]-peaks[i][0])
    return steptimes

def _steptime_avg_std(peaks):
    """Calculate the average and standard deviation for the time between steps

    Keyword arguments:
    peaks -- List of times of peaks
        peaks = accproc.detectPeaksGCDC()

    source: HowToExtractPeaks.html
    """
    steptimpes = _getStepTimes(peaks)
    return np.mean(steptimes), np.std(steptimes)

def _max_psd(signal):
    """Index and value of the maximum PSD value (psd from matplotlib)

    Keyword arguments:
    signal -- time series of a signal
    """
    psd = mlab.psd(signal)
    max_index = np.argsort(psd)[-1]
    return max_index, psd[max_index]

def _fourierdomain_peak(signal):
    """Index and value of the highest peak in the fourier domain of a signal."""
    fft = np.fft(signal)
    peak_index = np.argsort(fft)[-1]
    return peak_index, fft[peak_index]

def _enumerate_reversed(L):
    """Equivalent to: reversed(list(enumerate(L))), but avoids copying L"""
    for index in reversed(xrange(len(L))):
        yield index, L[index]

def _filter_single_series(series, fraction=0.2, characteristic=np.max,
                            cmp=operator.lt):
    """Returns two indices that would filter the series in a slice operation.

    Returns the first and last index where the value of the series
    satifies 'cmp' with regard to a 'fraction' of the 'characteristic'
    applied to the series, e.g.:
        cmp(fraction*characteristic(series), valueofseries)

    Keyword arguments:
    series -- the series to filter
    fraction -- fraction of the characteristic the value of the
                series must satisfy
                default: 20%
    characteristic -- a function that characterises a series as something that
                        can be 'cmp'ed to a value of the series
                        default: np.max
    cmp -- a comparison function
            default: operator.lt (python's function form of '<')
    """
    beginning = 0
    ending = len(series)
    criterion = fraction * characteristic(series)
    for index, val in enumerate(series):
        if cmp(criterion,val):
            beginning = index
            break
    for index, val in _enumerate_reversed(series):
        if cmp(criterion,val):
            ending = index
            break
    return beginning, ending

def _filter_series(*series, **kwargs):
    """Smallest and largest index that would filter each of the series.
    
    Keyword arguments:
    accepts the same keyword arguments as _filter_single_series
    defaults:
        fraction = 0.2
        characteristic = np.max
        cmp = operator.lt
    """
    fraction = kwargs.pop('fraction', 0.2)
    characteristic = kwargs.pop('characteristic', np.max)
    cmp = kwargs.pop('cmp', operator.lt)

    beginnings = []
    endings = []
    for single_series in series:
        b, e = _filter_single_series(single_series,
                                        fraction,
                                        characteristic,
                                        cmp)
        beginning.append(b)
        ending.append(e)
    return np.max(beginnings), np.min(endings)


def derive(runnerfile, nb_windows=1, window_size=256, window_shift=1):
    """Derive a number of features from triaxial accelerometer measurements.

    Requires a csv file that can be read by readGCDCFormat() as defined in
    classifyrunner.accproc

    Keyword arguments:
    runnerfile -- path to csv file
    """
    data = ap.preprocessGCDC(ap.readGCDCFormat(runnerfile))
    # Filter data (beginning, end; speed up,down) and limit length of series
    Ay = data['Ay'].values
    beginning, ending = _filter_single_series(Ay)
    print data
    data = data[beginning:ending]
    print data
    data.Ax.plot()
    plt.show()

if __name__ == "__main__":
    derive("data/Runs/Ann/enkel/DATA-005.CSV")
###Ax_max = ap.detectPeaksGCDC(data,
###                            columnname="Ax",
###                            detection={'delta':0.7},
###                            smooth={'type':'hilbert,butter',
###                                    'fcol':6,
###                                    'correct':True},
###                            plot=plot,
###                            verbose=True)
###Az_max = ap.detectPeaksGCDC(data,
###                            columnname="Az",
###                            detection={'lookahead':40, 'delta':0.4},
###                            smooth={'type':'hilbert,butter',
###                                    'fcol':10,
###                                    'correct':True},
###                            plot=plot,
###                            verbose=True)
###At_max = ap.detectPeaksGCDC(data,
###                            columnname="Atotal",
###                            detection={'type':'simple',
###                                        'lookahead':50,
###                                        'delta':1.0},
###                            smooth={'type':'hilbert,butter',
###                                    'fcol':5,
###                                    'correct':True,
###                                    'dist':0.1},
###                            plot=plot,
###                            verbose=True)
###Ax_xm,Ax_ym = zip(*Ax_max)

