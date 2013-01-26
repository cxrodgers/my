"""Catchall module within the catchall module for really one-off stuff."""

import numpy as np

def parse_folded_by_block(lb_folded, pb_folded, start_trial=1, last_trial=None,
    session_name=None):
    """Parses Folded into each block
    
    Using the known block structure (80 trials LB, 80 trials PB, etc) 
    and the trial labels in the folded, split the counts into each 
    sequential block.
    
    lb_folded, pb_folded : Folded with `label` attribute set with trial number
    start_trial : where to start counting up by 80
    last_trial : last trial to include, inclusive
        if None, use the max trial in either set of labels
    session_name : optional. If one of the known munged sessions, it will
        autoset start_trial
    
    Returns: counts_by_block
    A list of arrays, always beginning with LB, eg LB1, PB1, LB2, PB2...
    """
    # Override start trial for some munged sessions
    if session_name in ['YT6A_120201_behaving', 'CR24A_121019_001_behaving']:
        start_trial = 161
    
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
