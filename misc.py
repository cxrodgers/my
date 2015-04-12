"""Catchall module within the catchall module for really one-off stuff."""

import numpy as np
import warnings
import matplotlib.mlab as mlab
import matplotlib # for spectrogrammer
import os
import re
import datetime
import glob
import my

## Deprecated stuff
from my.video import OutOfFrames

def frame_dump(*args, **kwargs):
    warnings.warn("use my.video instead of my.misc", stacklevel=2)
    return my.video.frame_dump(*args, **kwargs)

def frame_dump_pipe(*args, **kwargs):
    warnings.warn("use my.video instead of my.misc", stacklevel=2)
    return my.video.get_frame(*args, **kwargs)
    
def process_chunks_of_video(*args, **kwargs):
    warnings.warn("use my.video instead of my.misc", stacklevel=2)
    return my.video.process_chunks_of_video(*args, **kwargs)

def get_video_aspect(*args, **kwargs):
    warnings.warn("use my.video instead of my.misc", stacklevel=2)
    return my.video.get_video_aspect(*args, **kwargs)

def get_video_duration(*args, **kwargs):
    warnings.warn("use my.video instead of my.misc", stacklevel=2)
    return my.video.get_video_duration(*args, **kwargs)
##

def globjoin(dirname, pattern, normalize=True):
    """Join dirname to pattern, and glob it
    
    If normalize: calls os.path.abspath on every result
    """
    res = glob.glob(os.path.join(dirname, pattern))
    if normalize:
        res = map(os.path.abspath, res)
    return res

def time_of_file(filename, fmt='%Y%m%d%H%M%S'):
    """Return the modification time of the file as a datetime.
    
    If fmt is not None: apply strftime(fmt) and return the string
    """
    dt = datetime.datetime.fromtimestamp(os.path.getmtime(filename))
    
    if fmt is None:
        return dt
    else:
        return dt.strftime(fmt)


class Spectrogrammer:
    """Turns a waveform into a spectrogram"""
    def __init__(self, NFFT=256, downsample_ratio=1, new_bin_width_sec=None,
        max_freq=None, min_freq=None, Fs=1.0, noverlap=None, normalization=0,
        detrend=mlab.detrend_mean, **kwargs):
        """Object to turn waveforms into spectrograms.
        
        This is a wrapper around mlab.specgram. What this object provides
        is slightly more intelligent parameter choice, and a nicer way
        to trade off resolution in frequency and time. It also remembers
        parameter choices, so that the same object can be used to batch
        analyze a bunch of waveforms using the `transform` method.
        
        Arguments passed to mlab.specgram
        ----------------------------------
        NFFT - number of points used in each segment
            Determines the number of frequency bins, which will be
            NFFT / 2 before stripping out those outside min_freq and max_freq
        
        noverlap - int, number of samples of overlap between segments
            Default is NFFT / 2
        
        Fs - sampling rate
        
        detrend - detrend each segment before FFT
            Default is to remove the mean (DC component)
        
        **kwargs - anything else you want to pass to mlab.specgram
        
        
        Other arguments
        ---------------
        downsample_ratio - int, amount to downsample in time
            After all other calculations are done, the temporal resolution
        
        new_bin_width_sec - float, target temporal resolution
            The returned spectrogram will have a temporal resolution as
            close to this as possible.
            If this is specified, then the downsample_ratio is adjusted
            as necessary to achieve it. If noverlap is left as default,
            it will try 50% first and then 0, to achieve the desired resolution.
            If it is not possible to achieve within a factor of 2 of this
            resolution, a warning is issued.
        
        normalization - the power in each frequency bin is multiplied by
            the frequency raised to this power.
            0 means do nothing.
            1 means that 1/f noise becomes white.
        
        min_freq, max_freq - discard frequencies outside of this range
        
        
        Returns
        -------
        Pxx - 2d array of power in dB. Shape (n_freq_bins, n_time_bins)
            May contain -np.inf where the power was exactly zero.
        
        freqs - 1d array of frequency bins
        
        t - 1d array of times
        
        
        Theory
        ------
        The fundamental tradeoff is between time and frequency resolution and
        is set by NFFT.
        
        For instance, consider a 2-second signal, sampled at 1024Hz, chosen
        such that the number of samples is 2048 = 2**11.
        *   If NFFT is 2048, you will have 1024 frequency bins (spaced 
            between 0KHz and 0.512KHz) and 1 time bin. 
            This is a simple windowed FFT**2, with the redundant negative
            frequencies discarded since the waveform is real.
            Note that the phase information is lost.
        *   If NFFT is 256, you will have 128 frequency bins and 8 time bins.
        *   If NFFT is 16, you will have 8 freqency bins and 128 time bins.
        
        In each case the FFT-induced trade-off is:
            n_freq_bins * n_time_bins_per_s = Fs / 2
            n_freq_bins = NFFT / 2
        
        So far, using only NFFT, we have traded off time resolution for
        frequency resolution. We can achieve greater noise reduction with
        appropriate choice of noverlap and downsample_ratio. The PSD
        function achieves this by using overlapping segments, then averaging
        the FFT of each segment. The use of noverlap in mlab.specgram is 
        a bit of a misnomer, since no temporal averaging occurs there!
        But this object can reinstate this temporal averaging.
        
        For our signal above, if our desired temporal resolution is 64Hz,
        that is, 128 samples total, and NFFT is 16, we have a choice.
        *   noverlap = 0. Non-overlapping segments. As above, 8 frequency
            bins and 128 time bins. No averaging
        *   noverlap = 64. 50% overlap. Now we will get 256 time bins.
            We can then average together each pair of adjacent bins
            by downsampling, theoretically reducing the noise. Note that
            this will be a biased estimate since the overlapping windows
            are not redundant.
        *   noverlap = 127. Maximal overlap. Now we will get about 2048 bins,
            which we can then downsample by 128 times to get our desired
            time resolution.
        
        The trade-off is now:
            overlap_factor = (NFFT - overlap) / NFFT
            n_freq_bins * n_time_bins_per_s * overlap_factor = Fs / downsample_ratio / 2
        
        Since we always do the smoothing in the time domain, n_freq bins = NFFT / 2
        and the tradeoff becomes
            n_time_bins_per_s = Fs / downsample_ratio / (NFFT - overlap)
        
        That is, to increase the time resolution, we can:
            * Decrease the frequency resolution (NFFT)
            * Increase the overlap, up to a maximum of NFFT - 1
              This is a sort of spurious improvement because adjacent windows
              are highly correlated.
            * Decrease the downsample_ratio (less averaging)
        
        To decrease noise, we can:
            * Decrease the frequency resolution (NFFT)
            * Increase the downsample_ratio (more averaging, fewer timepoints)
        
        How to choose the overlap, or the downsample ratio? In general,
        50% overlap seems good, since we'd like to use some averaging, but
        we get limited benefit from averaging many redundant samples.        
        
        This object tries for 50% overlap and adjusts the downsample_ratio
        (averaging) to achieve the requested temporal resolution. If this is
        not possible, then no temporal averaging is done (just like mlab.specgram)
        and the overlap is increased as necessary to achieve the requested
        temporal resolution.
        """
        self.downsample_ratio = downsample_ratio # until set otherwise
        
        # figure out downsample_ratio
        if new_bin_width_sec is not None:
            # Set noverlap to default
            if noverlap is None:
                # Try to do it with 50% overlap
                noverlap = NFFT / 2
            
            # Calculate downsample_ratio to achieve this
            self.downsample_ratio = \
                Fs * new_bin_width_sec / float(NFFT - noverlap)
            
            # If this is not achievable, then try again with minimal downsampling
            if np.rint(self.downsample_ratio).astype(np.int) < 1:
                self.downsample_ratio = 1
                noverlap = np.rint(NFFT - Fs * new_bin_width_sec).astype(np.int)
            
        # Convert to nearest int and test if possible
        self.downsample_ratio = np.rint(self.downsample_ratio).astype(np.int)        
        if self.downsample_ratio == 0:
            print "requested temporal resolution too high, using maximum"
            self.downsample_ratio = 1
    
        # Default value for noverlap if still None
        if noverlap is None:
            noverlap = NFFT / 2
        self.noverlap = noverlap
        
        # store other defaults
        self.NFFT = NFFT
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.Fs = Fs
        self.normalization = normalization
        self.detrend = detrend
        self.specgram_kwargs = kwargs

    
    def transform(self, waveform):
        """Converts a waveform to a suitable spectrogram.
        
        Removes high and low frequencies, rebins in time (via median)
        to reduce data size. Returned times are the midpoints of the new bins.
        
        Returns:  Pxx, freqs, t    
        Pxx is an array of dB power of the shape (len(freqs), len(t)).
        It will be real but may contain -infs due to log10
        """
        # For now use NFFT of 256 to get appropriately wide freq bands, then
        # downsample in time
        Pxx, freqs, t = mlab.specgram(waveform, NFFT=self.NFFT, 
            noverlap=self.noverlap, Fs=self.Fs, detrend=self.detrend, 
            **self.specgram_kwargs)
        
        # Apply the normalization
        Pxx = Pxx * np.tile(freqs[:, np.newaxis] ** self.normalization, 
            (1, Pxx.shape[1]))

        # strip out unused frequencies
        if self.max_freq is not None:
            Pxx = Pxx[freqs < self.max_freq, :]
            freqs = freqs[freqs < self.max_freq]
        if self.min_freq is not None:
            Pxx = Pxx[freqs > self.min_freq, :]
            freqs = freqs[freqs > self.min_freq]

        # Rebin in size "downsample_ratio". If last bin is not full, discard.
        Pxx_rebinned = []
        t_rebinned = []
        for n in range(0, len(t) - self.downsample_ratio + 1, 
            self.downsample_ratio):
            Pxx_rebinned.append(
                np.median(Pxx[:, n:n+self.downsample_ratio], axis=1).flatten())
            t_rebinned.append(
                np.mean(t[n:n+self.downsample_ratio]))

        # Convert to arrays
        Pxx_rebinned_a = np.transpose(np.array(Pxx_rebinned))
        t_rebinned_a = np.array(t_rebinned)

        # log it and deal with infs
        Pxx_rebinned_a_log = -np.inf * np.ones_like(Pxx_rebinned_a)
        Pxx_rebinned_a_log[np.nonzero(Pxx_rebinned_a)] = \
            10 * np.log10(Pxx_rebinned_a[np.nonzero(Pxx_rebinned_a)])


        self.freqs = freqs
        self.t = t_rebinned_a
        return Pxx_rebinned_a_log, freqs, t_rebinned_a

def fix_pandas_display_width(dw=0):
    """Sets display width to 0 (auto) or other"""
    import pandas
    pandas.set_option('display.width', dw)

class UniquenessError(Exception):
    pass

def only_one(l):
    """Returns the only value in l, or l itself if non-iterable.
    
    Compare 'unique_or_error', which allows multiple identical etnries.
    """
    # listify
    if not hasattr(l, '__len__'):
        l = [l]
    
    # check length
    if len(l) != 1:
        raise UniquenessError("must contain exactly one value; instead: %r" % l)
    
    # return entry
    return l[0]

def unique_or_error(a):
    """Asserts that `a` contains only one unique value and returns it
    
    Compare 'only_one' which does not allow repeats.
    """    
    u = np.unique(np.asarray(a))
    if len(u) == 0:
        raise UniquenessError("no unique values found, should be one")
    if len(u) > 1:
        raise UniquenessError("%d unique values found, should be one" % len(u))
    else:
        return u[0]

def printnow(s):
    """Write string to stdout and flush immediately"""
    import sys
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()

def get_file_time(filename, human=False):
    import time
    # Get modification time
    res = os.path.getmtime(filename)
    
    # Convert to human-readable
    if human:
        res = time.ctime(res)
    return res

def pickle_load(filename):
    import cPickle
    with file(filename) as fi:
        res = cPickle.load(fi)
    return res

def pickle_dump(obj, filename):
    import cPickle
    with file(filename, 'w') as fi:
        cPickle.dump(obj, fi)

def invert_linear_poly(p):
    """Helper function for inverting fit.coeffs"""
    return np.array([1, -p[1]]).astype(np.float) / p[0]

def apply_and_filter_by_regex(pattern, list_of_strings, sort=True):
    """Apply regex pattern to each string and return result.
    
    Non-matches are ignored.
    If multiple matches, the first is returned.
    """
    res = []
    for s in list_of_strings:
        m = re.match(pattern, s)
        if m is None:
            continue
        else:
            res.append(m.groups()[0])
    if sort:
        return sorted(res)
    else:
        return res

def regex_filter(pattern, list_of_strings):
    """Apply regex pattern to each string and return those that match.
    
    See also regex_capture
    """
    return [s for s in list_of_strings if re.match(pattern, s) is not None]

def regex_capture(pattern, list_of_strings, take_index=0):
    """Apply regex pattern to each string and return a captured group.

    Same as old apply_and_filter_by_regex, but without the sorting.
    See also regex_filter. This will match that order.
    """
    # Apply filter to each string
    res_l = []
    for s in list_of_strings:
        m = re.match(pattern, s)
        
        # Append the capture, if any
        if m is not None:
            res_l.append(m.groups()[take_index])
    
    return res_l
    

def rint(arr):
    """Round with rint and cast to int
    
    If `arr` contains NaN, casting it to int causes a spuriously negative
    number, because NaN cannot be an int. In this case we raise ValueError.
    """
    if np.any(np.isnan(np.asarray(arr))):
        raise ValueError("cannot convert arrays containing NaN to int")
    return np.rint(arr).astype(np.int)

def is_nonstring_iter(val):
    """Check if the input is iterable, but not a string.
    
    Recently changed this to work for Unicode. 
    This should catch a subset of the old way, because previously Unicode
    strings caused this to return True, but now they should return False.
    
    Will print a warning if this is not the case.
    """
    # Old way
    res1 = hasattr(val, '__len__') and not isinstance(val, str)
    
    # New way
    res2 = hasattr(val, '__len__') and not isinstance(val, basestring)
    
    if res2 and not res1:
        print "warning: check is_nonstring_iter"
    
    return res2

def pick(df, isnotnull=None, **kwargs):
    """Function to pick row indices from DataFrame.
    
    Copied from kkpandas
    
    This method provides a nicer interface to choose rows from a DataFrame
    that satisfy specified constraints on the columns.
    
    isnotnull : column name, or list of column names, that should not be null.
        See pandas.isnull for a defintion of null
    
    All additional kwargs are interpreted as {column_name: acceptable_values}.
    For each column_name, acceptable_values in kwargs.items():
        The returned indices into column_name must contain one of the items
        in acceptable_values.
    
    If acceptable_values is None, then that test is skipped.
        Note that this means there is currently no way to select rows that
        ARE none in some column.
    
    If acceptable_values is a single string or value (instead of a list), 
    then the returned rows must contain that single string or value.
    
    TODO:
    add flags for string behavior, AND/OR behavior, error if item not found,
    return unique, ....
    """
    msk = np.ones(len(df), dtype=np.bool)
    for key, val in kwargs.items():
        if val is None:
            continue
        elif is_nonstring_iter(val):
            msk &= df[key].isin(val)
        else:
            msk &= (df[key] == val)
    
    if isnotnull is not None:
        # Edge case
        if not is_nonstring_iter(isnotnull):
            isnotnull = [isnotnull]
        
        # Filter by not null
        for key in isnotnull:
            msk &= -pandas.isnull(df[key])

    return df.index[msk]

def pick_rows(df, **kwargs):
    """Returns sliced DataFrame based on indexes from pick"""
    return df.ix[pick(df, **kwargs)]

def no_warn_rs():
    warnings.filterwarnings('ignore', module='ns5_process.RecordingSession$')    

def parse_by_block(lb_counts, pb_counts, lb_trial_numbers, pb_trial_numbers,
    start_trial=None, last_trial=None, session_name=None):
    """Parses counts into each block (with corresponding trial numbers)
    
    Using the known block structure (80 trials LB, 80 trials PB, etc) 
    and the trial labels in the folded, split the counts into each 
    sequential block.
    
    parse_folded_by_block could almost be reimplemented with this function
    if only it accepted __getslice__[trial_number]
    Instead the code is almost duplicated.
    
    This one is overridden because it starts at 157:
        YT6A_120201_behaving, start at 161
    
    lb_counts, pb_counts : Array-like, list of counts, one per trial
    lb_trial_numbers, pb_trial_numbers : Array-like, same as counts, but
        contianing trial numbers.
    start_trial : where to start counting up by 80
        if None, auto-get from session_db and unit_db (or just use 1 if
        session name not provided)
    last_trial : last trial to include, inclusive
        if None, use the max trial in either set of labels
    session_name : if start_trial is None and you specify this, it will
        auto grab start_trial from session_db
    
    Returns: counts_by_block
    A list of arrays, always beginning with LB, eg LB1, PB1, LB2, PB2...
    """
    # Auto-get first trial
    if start_trial is None:
        if session_name is None:
            start_trial = 1
        elif session_name == 'YT6A_120201_behaving':
            # Forcible override
            start_trial = 161
        else:
            import my.dataload
            session_db = my.dataload.getstarted()['session_db']
            first_trial = int(round(session_db['first_trial'][session_name]))
            # Convert to beginning of first LBPB with any trials
            # The first block might be quite short
            # Change the final +1 to +161 to start at the first full block
            start_trial = ((first_trial - 1) / 160) * 160 + 1
    
    # Arrayify
    lb_counts = np.asarray(lb_counts)
    pb_counts = np.asarray(pb_counts)
    lb_trial_numbers = np.asarray(lb_trial_numbers)
    pb_trial_numbers = np.asarray(pb_trial_numbers)
    
    # Where to stop putting trials into blocks    
    if last_trial is None:
        last_trial = np.max([lb_trial_numbers.max(), pb_trial_numbers.max()])
    
    # Initialize return variable
    res_by_block = []
    
    # Parse by block
    for block_start in range(start_trial, last_trial + 1, 80):
        # Counts from lb in this block
        lb_this_block_msk = (
            (lb_trial_numbers >= block_start) &
            (lb_trial_numbers < block_start + 80))
        lb_this_block = lb_counts[lb_this_block_msk]
        
        # Counts from pb in this block
        pb_this_block_msk = (
            (pb_trial_numbers >= block_start) &
            (pb_trial_numbers < block_start + 80))
        pb_this_block = pb_counts[pb_this_block_msk]
        
        # Error check
        if np.mod(block_start - start_trial, 160) == 0:
            # Should be in an LB block
            assert len(pb_this_block) == 0
            if len(lb_this_block) == 0:
                print "warning: no trials around trial %d" % block_start
            res_by_block.append(lb_this_block)
        else:
            # Should be in a PB block
            assert len(lb_this_block) == 0
            if len(pb_this_block) == 0:
                print "warning: no trials around trial %d" % block_start            
            res_by_block.append(pb_this_block)
    
    # Error check that all counts were included and ordering maintained
    assert np.all(np.concatenate(res_by_block[::2]) == 
        lb_counts[lb_trial_numbers >= start_trial])
    assert np.all(np.concatenate(res_by_block[1::2]) == 
        pb_counts[pb_trial_numbers >= start_trial])
    
    return res_by_block 

def parse_folded_by_block(lb_folded, pb_folded, start_trial=1, last_trial=None,
    session_name=None):
    """Parses Folded into each block
    
    parse_by_block is now more feature-ful
    TODO: reimplement parse_by_block to just return trial numbers, then
    this function can wrap that and use the trial numbers to slice the foldeds
    
    Using the known block structure (80 trials LB, 80 trials PB, etc) 
    and the trial labels in the folded, split the counts into each 
    sequential block.
    
    lb_folded, pb_folded : Folded with `label` attribute set with trial number
    start_trial : where to start counting up by 80
    last_trial : last trial to include, inclusive
        if None, use the max trial in either set of labels
    session_name : no longer used. should make this load trials_info and
        grab first trial, if anything.
    
    Returns: counts_by_block
    A list of arrays, always beginning with LB, eg LB1, PB1, LB2, PB2...
    """
    # Override start trial for some munged sessions
    #~ if session_name in ['YT6A_120201_behaving', 'CR24A_121019_001_behaving']:
        #~ print "overriding start trial"
        #~ start_trial = 161
    
    # Where to stop putting trials into blocks    
    if last_trial is None:
        last_trial = np.max([lb_folded.labels.max(), pb_folded.labels.max()])
    
    # Initialize return variable
    res_by_block = []
    
    # Parse by block
    for block_start in range(start_trial, last_trial + 1, 80):
        # Counts from lb in this block
        lb_this_block_msk = (
            (lb_folded.labels >= block_start) &
            (lb_folded.labels < block_start + 80))
        lb_this_block = lb_folded.get_slice(lb_this_block_msk)
        
        # Counts from pb in this block
        pb_this_block_msk = (
            (pb_folded.labels >= block_start) &
            (pb_folded.labels < block_start + 80))
        pb_this_block = pb_folded.get_slice(pb_this_block_msk)
        
        # Error check
        if np.mod(block_start - start_trial, 160) == 0:
            # Should be in an LB block
            assert len(pb_this_block) == 0
            #assert len(lb_this_block) > 0
            if len(lb_this_block) == 0:
                print "warning: no trials around trial %d" % block_start
            res_by_block.append(lb_this_block)
        else:
            # Should be in a PB block
            assert len(lb_this_block) == 0
            #assert len(pb_this_block) > 0
            if len(pb_this_block) == 0:
                print "warning: no trials around trial %d" % block_start            
            res_by_block.append(pb_this_block)
    
    return res_by_block 


def yoked_zscore(list_of_arrays, axis=1):
    """Concatenate arrays, z-score together, break apart again"""
    concatted = np.concatenate(list_of_arrays, axis=axis)
    means = np.mean(concatted, axis=axis)
    stdevs = np.std(concatted, axis=axis)
    
    res = []
    for arr in list_of_arrays:
        if axis == 1:
            res.append((arr - means[:, None]) / stdevs[:, None])
        elif axis == 0:
            res.append((arr - means[None, :]) / stdevs[None, :])
        else:
            raise ValueError("axis must be 0 or 1")
    return res


def gaussian_smooth(signal, gstd=100, glen=None, axis=1, **filtfilt_kwargs):
    """Smooth a signal with a Gaussian window
    
    signal : array-like, to be filtered
    gstd : standard deviation of Gaussian in samples (can be float)
    glen : half-width of (truncated) Gaussian
        Default is int(2.5 * gstd)
        If you are using padding (on by default) and the pad length which
        is a function of `glen` is longer than the data, you will get a
        smoothing error. Lower `glen` or lower `padlen`.
    axis : 0 or 1
        Default is to filter the columns of 2d data
    filtfilt_kwargs : other kwargs to pass to filtfilt
        padtype - 'odd', 'even', 'constant', None
            Default is 'odd', that is, continuing the signal at either end
            with odd symmetry
        padlen - int or None
            Default is None, which is 3 * max(len(signal), glen)
    
    NaNs will cause problems. You should probably interpolate them, using
    perhaps interp_nans in this module.
    """
    import scipy.signal
    
    # Defaults
    signal = np.asarray(signal)
    if glen is None:    
        glen = int(2.5 * gstd)
    
    # Incantation such that b[0] == 1.0
    b = scipy.signal.gaussian(glen * 2, gstd, sym=False)[glen:]
    b = b / b.sum()
    
    # Smooth
    if signal.ndim == 1:
        res = scipy.signal.filtfilt(b, [1], signal, **filtfilt_kwargs)
    elif signal.ndim == 2:
        if axis == 0:
            res = np.array([scipy.signal.filtfilt(b, [1], sig, **filtfilt_kwargs) 
                for sig in signal])
        elif axis == 1:
            res = np.array([scipy.signal.filtfilt(b, [1], sig, **filtfilt_kwargs) 
                for sig in signal.T]).T
        else:
            raise ValueError("axis must be 0 or 1")
    else:
        raise ValueError("signal must be 1d or 2d")
    
    return res

def interp_nans(signal, axis=1, left=None, right=None, dtype=np.float):
    """Replaces nans in signal by interpolation along axis
    
    signal : array-like, containing NaNs
    axis : 0 or 1
        Default is to interpolate along columns
    left, right : to be passed to interp
    dtype : Signal is first converted to this type, mainly to avoid
        conversion to np.object
    """
    # Convert to array
    res = np.asarray(signal, dtype=np.float).copy()

    # 1d or 2d behavior
    if res.ndim == 1:
        # Inner loop
        nan_mask = np.isnan(res)
        res[nan_mask] = np.interp(
            np.where(nan_mask)[0], # x-coordinates where we need a y
            np.where(~nan_mask)[0], # x-coordinates where we know y
            res[~nan_mask], # known y-coordinates
            left=left, right=right)
    elif res.ndim == 2:
        if axis == 0:
            res = np.array([
                interp_nans(sig, left=left, right=right, dtype=dtype)
                for sig in res])
        elif axis == 1:
            res = np.array([
                interp_nans(sig, left=left, right=right, dtype=dtype)
                for sig in res.T]).T
        else:
            raise ValueError("axis must be 0 or 1")
    else:
        raise ValueError("signal must be 1d or 2d")
    return res


# Correlation and coherence functions
def correlate(v0, v1, mode='valid', normalize=True, auto=False):
    """Wrapper around np.correlate to calculate the timepoints

    'full' : all possible overlaps, from last of first and beginning of
        second, to vice versa. Total length: 2*N - 1
    'same' : Slice out the central 'N' of 'full'. There will be one more
        negative than positive timepoint.
    'valid' : only overlaps where all of both arrays are included
    
    normalize: accounts for the amount of data in each bin
    auto: sets the center peak to zero
    
    Positive peaks (latter half of the array) mean that the second array
    leads the first array.
   
    """
    counts = np.correlate(v0, v1, mode=mode)
    
    if len(v0) != len(v1):
        raise ValueError('not tested')
    
    if mode == 'full':
        corrn = np.arange(-len(v0) + 1, len(v0), dtype=np.int)
    elif mode == 'same':
        corrn = np.arange(-len(v0) / 2, len(v0) - (len(v0) / 2), 
            dtype=np.int)
    else:
        raise ValueError('mode not tested')
    
    if normalize:
        counts = counts / (len(v0) - np.abs(corrn)).astype(np.float)
    
    if auto:
        counts[corrn == 0] = 0
    
    return counts, corrn

def binned_pair2cxy(binned0, binned1, Fs=1000., NFFT=256, noverlap=None,
    windw=mlab.window_hanning, detrend=mlab.detrend_mean, freq_high=100,
    average_over_trials=True):
    """Given 2d array of binned times, return Cxy
    
    Helper function to ensure faking goes smoothly
    binned: 2d array, trials on rows, timepoints on cols
        Keep trials lined up!
    rest : psd_kwargs. noverlap defaults to NFFT/2

    Trial averaging, if any, is done between calculating the spectra and
    normalizing them.

    Will Cxy each trial with psd_kwargs, then mean over trials, then slice out 
    frequencies below freq_high and return.
    """
    # Set up psd_kwargs
    if noverlap is None:
        noverlap = NFFT / 2
    psd_kwargs = {'Fs': Fs, 'NFFT': NFFT, 'noverlap': noverlap, 
        'detrend': detrend, 'window': windw}

    # Cxy each trial
    ppxx_l, ppyy_l, ppxy_l = [], [], []
    for row0, row1 in zip(binned0, binned1):
        ppxx, freqs = mlab.psd(row0, **psd_kwargs)
        ppyy, freqs = mlab.psd(row1, **psd_kwargs)
        ppxy, freqs = mlab.csd(row0, row1, **psd_kwargs)
        ppxx_l.append(ppxx); ppyy_l.append(ppyy); ppxy_l.append(ppxy)
    
    # Optionally mean over trials, then normalize
    S12 = np.real_if_close(np.array(ppxy_l))
    S1 = np.array(ppxx_l)
    S2 = np.array(ppyy_l)
    if average_over_trials:
        S12 = S12.mean(0)
        S1 = S1.mean(0)
        S2 = S2.mean(0)
    Cxy = S12 / np.sqrt(S1 * S2)
    
    # Truncate unnecessary frequencies
    if freq_high:
        topbin = np.where(freqs > freq_high)[0][0]
        freqs = freqs.T[1:topbin].T
        Cxy = Cxy.T[1:topbin].T
    return Cxy, freqs

def binned2pxx(binned, Fs=1000., NFFT=256, noverlap=None,
    windw=mlab.window_hanning, detrend=mlab.detrend_mean, freq_high=100):
    """Given 2d array of binned times, return Pxx
    
    Helper function to ensure faking goes smoothly
    binned: 2d array, trials on rows, timepoints on cols
    rest : psd_kwargs. noverlap defaults to NFFT/2
    
    Will Pxx each trial separately with psd_kwargs, then slice out 
    frequencies below freq_high and return.
    """
    # Set up psd_kwargs
    if noverlap is None:
        noverlap = NFFT / 2
    psd_kwargs = {'Fs': Fs, 'NFFT': NFFT, 'noverlap': noverlap, 
        'detrend': detrend, 'window': windw}    
    
    # Pxx each trial
    ppxx_l = []
    for row in binned:
        ppxx, freqs = mlab.psd(row, **psd_kwargs)
        ppxx_l.append(ppxx)
    
    # Truncate unnecessary frequencies
    if freq_high:
        topbin = np.where(freqs > freq_high)[0][0]
        freqs = freqs[1:topbin]
        Pxx = np.asarray(ppxx_l)[:, 1:topbin]       
    return Pxx, freqs


def sem(data, axis=None):
    """Standard error of the mean"""
    if axis is None:
        N = len(data)
    else:
        N = np.asarray(data).shape[axis]
    
    return np.std(np.asarray(data), axis) / np.sqrt(N)

