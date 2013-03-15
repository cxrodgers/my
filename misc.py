"""Catchall module within the catchall module for really one-off stuff."""

import numpy as np
import warnings

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
    b = b / np.sum(b**2)   
    
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
