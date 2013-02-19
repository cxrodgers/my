"""Methods for finding PSTH peaks of unknown timecourse"""

from stats import r_utest
import numpy as np


def test_vs_baseline(data, baseline_idxs, test_idxs, fix_float=10000):
    """Test rows of a dataset against a baseline set of rows.
    
    For instance the rows could be timepoints and the columns replicates.
    Which timepoints are significantly greater (or less than) a set of
    timepoints defined as baseline?
    
    This tests each row in the test set separately against the full baseline
    test set using Mann-Whitney U and returns the p-value of each test row,
    along with some other diagnostics.
    
    Parameters
    ----------
    data : array-like, replicates in columns
    baseline_idxs : rows in data to be treated as baseline set
    test_idxs : row in data to test against baseline
    fix_float : if not None or False, then the data is pre-multiplied by
        this and fixed to an integer. This prevents numerical differences
        of less than 1/fix_float from affecting the results. It might also
        be slightly faster.
    
    Returns
    -------
    u, p, auroc, ref_counts, all_test_counts:        
        u : array of length N containing U statistic for each timepoint (row)
            Large values indicate x > y
        p : array of length N containing two-tailed p-value for each row
        auroc : array of length N containg area under region of convergence
            for each row
            Values > 0.5 indicate x > y
    """
    # Check it's an array
    data = np.asarray(data)
    
    if fix_float:
        data = (data * fix_float).astype(np.int)

    # get the baseline counts (note these may be non-integer due to smoothing)
    try:
        refdata = data[baseline_idxs].flatten()
    except IndexError:
        raise ValueError("baseline indexes are not valid row indexes")

    # test each time bin separately
    all_test_counts, all_u, all_p, all_auroc, all_dir = [], [], [], [], []
    for idx in test_idxs:
        # get this test set
        try:
            testdata = data[idx].flatten()
        except IndexError:
            raise ValueError("provided test index not valid for provided data")
        
        # test it (the ordering is important: testdata > refdata => auroc > .5)
        utest_res = r_utest(testdata, refdata)
        U, p, auroc = utest_res['U'], utest_res['p'], utest_res['auroc']
        
        # store results
        all_u.append(U)
        all_p.append(p)
        all_auroc.append(auroc)

    # convert to array
    all_u = np.asarray(all_u) # by timepoint
    all_p = np.asarray(all_p) # by timepoint
    all_auroc = np.asarray(all_auroc) # by timepoint

    return all_u, all_p, all_auroc



def define_onset_window(auroc, p, min_duration=2, max_stop_idx=None, 
    min_start_idx=0, drop_truncated=True):
    """Given AUROC and p-value, identify start and stop of putative peak.
    
    Define this as extending from first significant positive bin, till
    the first nonsignif bin (or first significantly negative bin).
    
    Specifically:
    1)  We consider all bins >= min_start_idx with signif pos peak as
        possible peak starts. If nothing to consider, msg = 'nothing'
    2)  We discard any peak start if the peak falls below significance
        in less than min_duration bins.
    3)  If none satisfy this constraint, msg='too brief'. If more than one,
        choose the earliest.
    
    # Remove these next two checks and let upstream user decide
    4)  If that peak lasts to the end of the testing window and
        drop_truncated is true, then msg = 'lasts to end'
    5)  If that peak lasts past max_stop_idx, then msg = 'lasts too long'
    6)  Otherwise msg = 'good' and the peak intervals are stored.
    
    Returns: dict with following keys
        start_idx, stop_idx: indexes into masked (half-open interval)
        peak_found : unless True, no peak was found, and start_idx and
            stop_idx are uninterpretable.
        msg : string as stated above
    """
    # Default values for returned variables
    start_idx, stop_idx, peak_found, msg = 0, 0, False, 'nothing'
    
    # Find where it started (after min_start_idx)
    masked = (auroc > .5) & (p < .05)
    start_candidates = np.where(masked)[0]
    
    # Mask out ones before min_start_idx
    start_candidates = start_candidates[start_candidates >= min_start_idx]
    
    # Mask out ones that are too close to the end
    start_candidates = start_candidates[
        start_candidates <= len(auroc) - min_duration]
    
    # Iterate over start candidates to find ones that satisfy the constraints
    for start_candidate in start_candidates:
        if masked[start_candidate:start_candidate + min_duration].all():
            peak_found = True
            start_idx = start_candidate
            msg = 'good'
            break
        else:
            msg = 'too brief'
    
    # If a peak was found, find the end of it
    if peak_found:
        stop_candidates = np.where(~masked)[0]
        stop_candidates = stop_candidates[stop_candidates > start_idx]
        try:
            stop_idx = stop_candidates[0]
        except IndexError:
            stop_idx = len(auroc) - 1
        assert masked[start_idx:stop_idx].all()
    
    # Some final checks
    if drop_truncated and stop_idx == len(auroc) - 1:
        #start_idx, stop_idx, peak_found = 0, 0, False
        msg = 'lasts to end'
    elif max_stop_idx and stop_idx > max_stop_idx:
        #start_idx, stop_idx, peak_found = 0, 0, False
        msg = 'lasts too long'
    
    return {'start_idx': start_idx, 'stop_idx': stop_idx, 
        'peak_found': peak_found, 'msg': msg}


def plot_peaks(peak_intervals, onset_results, tested_t, traces_per_ax=4):
    """Given results in peak_intervals and onset_results, plot the peaks
    
    onset_results : DataFrame, indexed by ulabel
        Each row comes from test_vs_baseline
    
    peak_intervals : DataFrame, indexed by ulabel
        Each row is the identified peak indices from define_onset_window
    
    tested_t : time indices of arrays like `auroc` in onset_results
    traces_per_ax : int
        Each type of ulabel is plotted in its own figure (eg nothing,
        too broad, etc). Each figure has enough subplots s.t. there are no
        more than `traces_per_ax` traces in each one.
    """
    import matplotlib.pyplot as plt
    
    # Plot the results of window detection, separately by type of peak
    gobj = peak_intervals.groupby('msg')
    for msg, sub_PI in gobj:
        # Divide figure into enough subplots
        n_traces = len(sub_PI)
        n_subplots = int(np.ceil(n_traces / float(traces_per_ax)))
        n_rows = np.int(np.sqrt(n_subplots))
        n_cols = int(np.ceil(float(n_subplots) / n_rows))
        
        # Create figure and iterate over subplots
        f, axa = plt.subplots(n_rows, n_cols, squeeze=False)
        f.suptitle(msg)
        for nax, ax in enumerate(axa.flatten()):
            # Resize to make room for legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
            
            # Grab the ulabels for this subplot
            to_plot_start = nax * traces_per_ax
            to_plot_stop = np.min([to_plot_start + traces_per_ax, n_traces])
            ulabels = sub_PI.index[to_plot_start:to_plot_stop]
            if len(ulabels) == 0: break
            
            # Grab the auroc for these traces and plot it
            aurocs = np.array(list(onset_results['auroc'][ulabels].values))
            ax.plot(tested_t, aurocs.T)
            
            # Plot again, thicker where signif
            pvals = np.array(list(onset_results['p'][ulabels].values))
            masked_auroc = aurocs.copy()
            masked_auroc[pvals > .05] = np.nan
            ax.plot(tested_t, masked_auroc.T, 'kx', ms=8)
            
            # Plot stars on points in peak
            for nu, ulabel in enumerate(ulabels):
                # Peak params for this ulabel
                i1 = peak_intervals['start_idx'][ulabel]
                i2 = peak_intervals['stop_idx'][ulabel]
                pf = peak_intervals['peak_found'][ulabel]
                auroc = onset_results['auroc'][ulabel]
                if pf:
                    # Thicker line for peak
                    ax.plot(tested_t[i1:i2], auroc[i1:i2], lw=4, 
                        color=ax.get_lines()[nu].get_color())
                    # Stars on peak
                    ax.plot(tested_t[i1:i2], auroc[i1:i2], 'k+', ms=8, mew=2)
            
            # Legend outside bbox
            ax.legend(ulabels, bbox_to_anchor=(0.9,1.2), prop={'size':'x-small'})
