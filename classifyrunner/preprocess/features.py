import numpy as np
import matplotlib.mlab as mlab
import classifyrunner.accproc as ap

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
    return np.mean(_getStepTimes(peaks)), np.std(_getStepTimes(peaks))

def _psd(signal):
    """Calculate PSD by Welch's average periodogram method (matplotlib)

    Keyword arguments:
    signal -- time series of a signal
    """
    return mlab.psd(signal)

def _fourierdomain_peak(signal):
    """Index and value of the highest peak in the fourier domain of a signal."""
    fft = np.fft(signal)
    peak_index = np.argsort(fft)[-1]
    return peak_index, fft[peak_index]

def derive(runnerfile):
    """Derive a number of features from triaxial accelerometer measurements.

    Requires a csv file that can be read by readGCDCFormat() as defined in
    classifyrunner.accproc

    Keyword arguments:
    runnerfile -- path to csv file
    """
    data = ap.preprocessGCDC(ap.readGCDCFormat(runnerfile))
    # Filter data (beginning, end; speed up,down) and limit length of series

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

