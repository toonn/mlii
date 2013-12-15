import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import classifyrunner.accproc as ap
import operator

def _acceleration_avg_std(data, columname):
    """Calculate average and standard deviation for acceleration meassurements

    source: HowToExtractPeaks.html
    """
    return np.mean(data[columname].values), np.std(data[columname].values)

def _detect_peaks(data, columname):
    """Helper function to call accproc detectPeaksGCDC.

    detectPeaksGCDC is very fragile when working on short series because
    peakdetect returns an empty list of positive peaks.
    A solution would be to do peakdetection on the original data and then
    select those peaks in the same interval as the window on which we work now.
    """
    A_peaks = 0
    A_peaks = ap.detectPeaksGCDC(data, columnname=columname,
                                detection={'lookahead': 20,
                                            'delta': 0.1},
                                smooth={'type':'butter'},
                                plot=False,
                                verbose=False)
        # delta > 0.1 Causes peakdetect in accproc not to find any
        # positive peaks
    return A_peaks

def _peak_avg_std(data, columname):
    """Calculate the average and standard deviation for the peak accelerations

    source: HowToExtractPeaks.html
    """
    peaks = _detect_peaks(data, columname)
    xm, ym = zip(*peaks)
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

def _steptime_avg_std(data, columname):
    """Calculate the average and standard deviation for the time between steps

    source: HowToExtractPeaks.html
    """
    peaks = _detect_peaks(data, columname)
    steptimes = _getStepTimes(peaks)
    return np.mean(steptimes), np.std(steptimes)

def _max_psd(data, columname):
    """Frequency and value of the maximum PSD value (psd from matplotlib) """
    signal = data[columname].values
    packed_Pxx, freqs = mlab.psd(signal)
    Pxx = [val for row in packed_Pxx for val in row]
    max_index = np.argsort(Pxx)[-1]
    return freqs[max_index], Pxx[max_index]

def _fourierdomain_peak(data, columname):
    """Index and value of the highest peak in the fourier domain of a signal."""
    signal = data[columname].values
    fft = np.abs(np.fft.fft(signal))
    peak_index = np.argsort(fft)[-1]
    return peak_index, fft[peak_index]

def _peak_to_acc_ratios(data, columname):
    """Ratios of average and standard deviation between peaks and signal.
    
    This is an attempt to normalize peak magnitude for runners/terrain.
    """
    avg_acc, std_acc = _acceleration_avg_std(data, columname)
    avg_peak, std_peak = _peak_avg_std(data, columname)
    return (avg_peak/avg_acc), (std_peak/std_acc)

def _single_to_total_ratio(data, columname):
    """Ratio between avg and std acceleration and total acceleration.
    
    We assume more experienced runners are more efficient,
    making this ratio larger.
    """
    tot_avg, tot_std = _acceleration_avg_std(data, 'Atotal')
    single_avg, single_std = _acceleration_avg_std(data, columname)
    return (single_avg/tot_avg), (single_std/tot_std)

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
    kwargs -- accepts the same keyword arguments as _filter_single_series
        defaults:
            fraction = 0.2
            characteristic = np.max
            cmp = operator.lt
    """
    fraction = kwargs.pop('fraction', 0.2)
    characteristic = kwargs.pop('characteristic', np.max)
    cmp = kwargs.pop('cmp', operator.lt)

    min_length = np.Inf
    trimmed_series = []
    for single_series in series:
        b, e = _filter_single_series(single_series.Ay.values,
                                        fraction,
                                        characteristic,
                                        cmp)
        if e-b < min_length:
            min_length = e-b
        trimmed_series.append(single_series[b:e])
    print 'min_length: ', min_length
    return [eq_len_trim_ser[:min_length] for eq_len_trim_ser in trimmed_series]

def extract_features(data):
    """Extract features from (trimmed) triaxial accelerometer data."""
    separate_features = {}
    for ax in data.keys():
        axfeatures = []
        axfeatures.extend(_acceleration_avg_std(data, ax))
        axfeatures.extend(_peak_avg_std(data, ax))
        axfeatures.extend(_max_psd(data, ax))
        axfeatures.extend(_fourierdomain_peak(data, ax))
        axfeatures.extend(_peak_to_acc_ratios(data, ax))
        axfeatures.extend(_single_to_total_ratio(data, ax))
        separate_features[ax] = axfeatures

    # all avg's, then all std's, then all peak_avg's, etc.
    features = [sep_fea[i]
                for i in xrange(len(separate_features.values()[0]))
                for sep_fea in separate_features.values()]
    features.extend(_steptime_avg_std(data, 'Atotal'))
    return features

def derive_single(runnerfile, nb_windows=1, window_size=256, window_shift=1):
    """Derive a number of features from triaxial accelerometer measurements.

    Requires a csv file that can be read by readGCDCFormat() as defined in
    classifyrunner.accproc

    Keyword arguments:
    runnerfile -- path to csv file
    """
    features_for_windows = []
    data = ap.preprocessGCDC(ap.readGCDCFormat(runnerfile))
    Ay = data['Ay'].values
    beginning, ending = _filter_single_series(Ay)
    data = data[beginning:ending]
    # This centers the series of windows around the center of the data.
    window_start = ((len(data['Ay'].values) / 2) -
                    ((window_size + nb_windows*window_shift) / 2))
    for nb in xrange(nb_windows):
        start = window_start + nb*window_shift
        window = data[start:(start + window_size)]
        features_for_windows.append(extract_features(window))

    return features_for_windows

def derive(*runnerfiles, **kwargs):
    """Derive a number of features from a collection of files.

    Features are derived for each of the files and concatenated.
    
    Keyword arguments:
    kwargs -- accepts the same keyword arguments as derive_single
    """
    nb_windows = kwargs.pop('nb_windows', 1)
    window_size = kwargs.pop('window_size', 256)
    window_shift = kwargs.pop('window_shift', 1)

    features_for_windows = []

    datalist = []
    Alist = []
    for runnerfile in runnerfiles:
        data = ap.preprocessGCDC(ap.readGCDCFormat(runnerfile))
        datalist.append(data)
    datalist = _filter_series(*datalist)
    # This centers the series of windows around the center of the data.
    window_start = ((len(datalist[0]) / 2) -
                    ((window_size + nb_windows*window_shift) / 2))
    data_grouped_windows = []
    for runnerdata in datalist:
        single_features_for_windows = []
        for nb in xrange(nb_windows):
            start = window_start + nb*window_shift
            window = runnerdata[start:(start + window_size)]
            single_features_for_windows.append(extract_features(window))
        data_grouped_windows.append(single_features_for_windows)

    window_grouped_datas = zip(*data_grouped_windows)
    for window_group in window_grouped_datas:
        feature_window = []
        for window in window_group:
            feature_window += window
        features_for_windows.append(feature_window)
    return features_for_windows

if __name__ == "__main__":
    print derive("data/Runs/Ann/enkel/DATA-001.CSV",
                    "data/Runs/Ann/heup/DATA-001.CSV")
