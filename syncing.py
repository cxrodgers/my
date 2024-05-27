"""Functions for detecting onsets and syncrhronizing events"""
# Moved from MCwatch.behavior.syncing

import numpy as np
import pandas

def extract_onsets_and_durations(lums, delta=30, diffsize=3, refrac=5,
    verbose=False, maximum_duration=100, meth=2):
    """Identify sudden, sustained increments in the signal `lums`.
    
    meth: int
        if 2, use extract_duration_of_onsets2 
            This was the default for a long time
            This is a "greedy algorithm". 
            It prioritizes earlier onsets / longer durations
        if 1, use extract_duration_of_onsets
            It prioritizes later onsets / shorter durations
        
        The difference occurs if we have onset1 (with no matching offset1),
        followed by matching (onset2, offset2). The greedy algorithm will
        prioritize the first onset, and match offset2 to onset1, so it has
        to drop onset2. The standard algorithm will prioritize the last onset
        before the upcoming offset (i.e., prioritize shorter duration).
    
    Algorithm
    1.  Take the diff of lums over a period of `diffsize`.
        In code, this is: lums[diffsize:] - lums[:-diffsize]
        Note that this means we cannot detect an onset before `diffsize`.
        Also note that this "smears" sudden onsets, so later we will always
        take the earliest point.
    2.  Threshold this signal to identify onsets (indexes above 
        threshold) and offsets (indexes below -threshold). Add `diffsize`
        to each onset and offset to account for the shift incurred in step 1.
    3.  Drop consecutive onsets that occur within `refrac` samples of
        each other. Repeat separately for offsets. This is done with
        the function `drop_refrac`. Because this is done separately, note
        that if two increments are separated by a brief return to baseline,
        the second increment will be completely ignored (not combined with
        the first one).
    4.  Convert the onsets and offsets into onsets and durations. This is
        done with the function `extract duration of onsets2`. This discards
        any onset without a matching offset.
    5.  Drop any matched onsets/offsets that exceed maximum_duration
    
    TODO: consider applying a boxcar of XXX frames first.
    
    Returns: onsets, durations
        onsets : array of the onset of each increment, in samples.
            This will be the first sample that includes the detectable
            increment, not the sample before it.
        durations : array of the duration of each increment, in samples
            Same length as onsets. This is "Pythonic", so if samples 10-12
            are elevated but 9 and 13 are not, the onset is 10 and the duration
            is 3.
    """
    # diff the sig over a period of diffsize
    diffsig = lums[diffsize:] - lums[:-diffsize]

    # Threshold and account for the shift
    onsets = np.where(diffsig > delta)[0] + diffsize
    offsets = np.where(diffsig < -delta)[0] + diffsize
    if verbose:
        print("initial onsets")
        print(onsets)
        print("initial offsets")
        print(offsets)
    
    # drop refractory onsets, offsets
    onsets2 = drop_refrac(onsets, refrac)
    offsets2 = drop_refrac(offsets, refrac)    
    if verbose:
        print("after dropping refractory violations: onsets")
        print(onsets2)
        print("after dropping refractory violations: offsets")
        print(offsets2)
    
    # Match onsets to offsets
    if meth == 1:
        remaining_onsets, durations = extract_duration_of_onsets(onsets2, offsets2)
    elif meth == 2:
        remaining_onsets, durations = extract_duration_of_onsets2(onsets2, offsets2)
    else:
        raise ValueError("unexpected meth {}, should be 1 or 2".format(meth))
    if verbose:
        print("after combining onsets and offsets: onsets-offsets-durations")
        print(np.array([remaining_onsets, remaining_onsets + durations, durations]).T)
    
    # apply maximum duration mask
    if maximum_duration is not None:
        max_dur_mask = durations <= maximum_duration
        remaining_onsets = remaining_onsets[max_dur_mask].copy()
        durations = durations[max_dur_mask].copy()
        
        if verbose:
            print("after applying max duration mask: onsets-offsets-durations")
            print(np.array([remaining_onsets, remaining_onsets + durations, durations]).T)


    return remaining_onsets, durations
    
def drop_refrac(arr, refrac):
    """Drop all values in arr after a refrac from an earlier val"""
    drop_mask = np.zeros_like(arr).astype(bool)
    for idx, val in enumerate(arr):
        drop_mask[(arr < val + refrac) & (arr > val)] = 1
    return arr[~drop_mask]

def extract_duration_of_onsets(onsets, offsets):
    """Extract duration of each onset.
    
    The duration is the time to the next offset. If there is another 
    intervening onset, then drop the first one.
    
    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    for idx, val in enumerate(onsets):
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Find upcoming onsets and skip if there is one before next offset
        upcoming_onsets = onsets[onsets > val]
        if len(upcoming_onsets) > 0 and upcoming_onsets[0] < next_offset:
            continue
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)    

    return np.asarray(onsets3), np.asarray(durations)

def extract_duration_of_onsets2(onsets, offsets):
    """Extract duration of each onset.
    
    Use a "greedy" algorithm. For each onset:
        * Assign it to the next offset
        * Drop any intervening onsets
        * Continue with the next onset

    Returns: remaining_onsets, durations
    """
    onsets3 = []
    durations = []
    
    if len(onsets) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    # This trigger will be set after each detected duration to mask out
    # subsequent onsets greedily
    onset_trigger = np.min(onsets) - 1
    
    # Iterate over onsets
    for idx, val in enumerate(onsets):
        # Skip onsets 
        if val < onset_trigger:
            continue
        
        # Find upcoming offsets and skip if none
        upcoming_offsets = offsets[offsets > val]
        if len(upcoming_offsets) == 0:
            continue
        next_offset = upcoming_offsets[0]
        
        # Store duration and this onset
        onsets3.append(val)
        durations.append(next_offset - val)
        
        # Save this trigger to skip subsequent onsets greedily
        onset_trigger = next_offset

    return np.asarray(onsets3), np.asarray(durations)

def longest_unique_fit(xdata, ydata, start_fitlen=3, ss_thresh=.0003,
    verbose=True, x_midslice_start=None, return_all_data=False,
    refit_data=False):
    """Find the longest consecutive string of fit points between x and y.

    We start by taking a slice from xdata of length `start_fitlen` 
    points. This slice is centered at `x_midslice_start` (by default,
    halfway through). We then take all possible contiguous slices of 
    the same length from `ydata`; fit each one to the slice from `xdata`;
    and calculate the best-fit sum-squared residual per data point. 
    
    (Technically seems we are fitting from Y to X.)
    
    If any slices have a per-point residual less than `ss_thresh`, then 
    increment the length of the fit and repeat. If none do, then return 
    the best fit for the previous iteration, or None if this is the first
    iteration.
    
    Usually it's best to begin with a small ss_thresh, because otherwise
    bad data points can get incorporated at the ends and progressively worsen
    the fit. If no fit can be found, try increasing ss_thresh, or specifying a
    different x_midslice_start. Note that it will break if the slice in
    xdata does not occur anywhere in ydata, so make sure that the midpoint
    of xdata is likely to be somewhere in ydata.

    xdata, ydata : unmatched data to be fit
    start_fitlen : length of the initial slice
    ss_thresh : threshold sum-squared residual per data point to count
        as an acceptable fit. These will be in the units of X.
    verbose : issue status messages
    x_midslice_start : the center of the data to take from `xdata`. 
        By default, this is the midpoint of `xdata`.
    return_all_data : boolean
        Return x_start, y_start, etc.
    refit_data : boolean, only matters if return_all_data = True
        Once the best xvy is determined, do a last refit on the maximum
        overlap of xdata and ydata.  Useful because normally execution
        stops when we run out of data (on either end) or when a bad point
        is reached. However, this will fail badly if either xdata or ydata
        contains spurious datapoints (i.e., concatenated from another 
        session).
    
    Returns: a linear polynomial fitting from Y to X.
        Or if return_all_data, also returns the start and stop indices
        into X and Y that match up. These are Pythonic (half-open).
    """
    # Choose the idx to start with in behavior
    fitlen = start_fitlen
    last_good_fitlen = 0
    if x_midslice_start is None:
        x_midslice_start = len(xdata) // 2
    keep_going = True
    best_fitpoly = None

    if verbose:
        print("begin with fitlen", fitlen)

    while keep_going:        
        # Slice out xdata
        chosen_idxs = xdata[x_midslice_start - fitlen:x_midslice_start + fitlen]
        
        # Check if we ran out of data
        if len(chosen_idxs) != fitlen * 2:
            if verbose:
                print("out of data, breaking")
            break
        if np.any(np.isnan(chosen_idxs)):
            if verbose:
                print("nan data, breaking")
            break

        # Find the best consecutive fit among onsets
        rec_l = []
        for idx in list(range(0, len(ydata) - len(chosen_idxs) + 1)):
            # The data to fit with
            test = ydata[idx:idx + len(chosen_idxs)]
            if np.any(np.isnan(test)):
                # This happens when the last data point in ydata is nan
                continue
            
            # fit
            fitpoly = np.polyfit(test, chosen_idxs, deg=1)
            fit_to_input = np.polyval(fitpoly, test)
            resids = chosen_idxs - fit_to_input
            ss = np.sum(resids ** 2)
            rec_l.append({'idx': idx, 'ss': ss, 'fitpoly': fitpoly})
        
        # Test if there were no fits to analyze
        if len(rec_l) == 0:
            keep_going = False
            if verbose:
                print("no fits to threshold, breaking")
            break
        
        # Look at results
        rdf = pandas.DataFrame.from_records(rec_l).set_index('idx').dropna()

        # Keep only those under thresh
        rdf = rdf[rdf['ss'] < ss_thresh * len(chosen_idxs)]    

        # If no fits, then quit
        if len(rdf) == 0:
            keep_going = False
            if verbose:
                print("no fits under threshold, breaking")
            break
        
        # Take the best fit
        best_index = rdf['ss'].idxmin()
        best_ss = rdf['ss'].min()
        best_fitpoly = rdf['fitpoly'].loc[best_index]
        if verbose:
            fmt = "fitlen=%d. best fit: x=%d, y=%d, xvy=%d, " \
                "ss=%0.3g, poly=%0.4f %0.4f"
            print(fmt % (fitlen, x_midslice_start - fitlen, best_index, 
                x_midslice_start - fitlen - best_index, 
                best_ss // len(chosen_idxs), best_fitpoly[0], best_fitpoly[1]))

        # Increase the size
        last_good_fitlen = fitlen
        fitlen = fitlen + 1    
    
    # Always return None if no fit found
    if best_fitpoly is None:
        return None
    
    if return_all_data:
        # Store results in dict
        fitdata = {
            'x_start': x_midslice_start - last_good_fitlen,
            'x_stop': x_midslice_start + last_good_fitlen,
            'y_start': best_index,
            'y_stop': best_index + last_good_fitlen * 2,
            'best_fitpoly': best_fitpoly,
            'xdata': xdata,
            'ydata': ydata,
        }            
        
        # Optionally refit to max overlap
        if refit_data:
            fitdata = refit_to_maximum_overlap(xdata, ydata, fitdata)
        
        return fitdata
    else:
        return best_fitpoly
