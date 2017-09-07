"""Neural data stuff"""
import OpenEphys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import my.OpenEphys
import MCwatch.behavior
import ArduFSM
import tables
import kkpandas
import Adapters

probename2ch_list = {
    'edge': [24, 26, 25, 29, 27, 31, 28, 1, 30, 5, 3, 0, 2, 4, 9, 11, 13, 
        15, 8, 23, 10, 22, 12, 21, 14, 20, 17, 18, 16, 19],
    'poly2': [5, 10, 30, 22, 1, 12, 28, 21, 31, 14, 27, 20, 29, 17, 18, 
        26, 16, 19, 23, 0, 15, 2, 13, 4, 11, 6, 9],
    }


def plot_each_channel(data, ax=None, n_range=None, ch_list=None, 
    downsample=1, exclude_ch_list=None, scaling_factor=.195,
    inter_ch_spacing=234, legend_t_offset=0.0, legend_y_offset=200,
    max_data_size=1e6, highpass=False, probename=None,
    spike_times=None, clusters=None, cluster_list=None, features_masks=None,
    cluster2color=None, legend_t_width=.010, apply_offset=None,
    plot_kwargs=None):
    """Plot a vertical stack of channels in the same ax.
    
    The given time range and channel range is extracted. Optionally it
    can be filtered at this point. 
    
    data : array, shape (N_timepoints, N_channels)
    
    ax : axis to plot into. 
        If None, a new figure and axis will be created
    
    n_range : tuple (n_start, n_stop)
        The index of the samples to include in the plot
        Default is (0, len(data))
    
    ch_list : a list of the channels to include, expressed as indices into
        the columns of `data`. This also determines the order in which they
        will be plotted (from top to bottom of the figure)
    
    exclude_ch_list : remove these channels from `ch_list`
    
    downsample : downsample by this factor
    
    scaling_factor : multiple by this
    
    inter_ch_spacing : channel centers are offset by this amount, in the
        same units as the data (after multiplication by scaling_factor)
    
    legend_t_offset, legend_y_offset, legend_t_width : where to plot legend
    
    max_data_size : sanity check, raise error rather than try to plot
        more than this amount of data
    
    plot_kwargs : a dict to pass to `plot`, containing e.g. linewidth
    """
    # Set up the channels to include
    if ch_list is None:
        if probename is None:
            ch_list = list(range(data.shape[1]))
        else:
            ch_list = probename2ch_list[probename]
    if exclude_ch_list is not None:
        for ch in exclude_ch_list:
            ch_list.remove(ch)

    # Set up data_range
    if n_range is None:
        n_range = (0, len(data))
    else:
        # Ensure int
        assert len(n_range) == 2
        n_range = tuple(map(int, n_range))
    
    # data range in seconds
    t = np.arange(n_range[0], n_range[1]) / 30000.
    t_ds = t[::downsample]

    # Grab the data that will actually be plotted
    got_size = len(t_ds) * len(ch_list)
    print "getting %g datapoints..." % got_size
    if len(t_ds) * len(ch_list) > max_data_size:
        raise ValueError("you requested %g datapoints" % got_size)
    got_data = data[n_range[0]:n_range[1], ch_list]
    got_data = got_data * scaling_factor
    
    if apply_offset:
        got_data = got_data + apply_offset
    
    if highpass:
        buttb, butta = scipy.signal.butter(3, 300./30e3, 'high')
        got_data = scipy.signal.filtfilt(buttb, butta, got_data, axis=0)
    
    got_data = got_data[::downsample]

    # Set up ax
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(5, 5))
    #~ f.patch.set_color('w')
    
    # Plot each channel
    for ncol, col in enumerate(got_data.T):
        y_offset = -inter_ch_spacing * ncol
        ax.plot(t_ds, col + y_offset, 'k', lw=.75)

    # Overplot spikes
    if spike_times is not None:
        # Find the spike times in this range
        spike_time_mask = (
            (spike_times >= n_range[0]) &
            (spike_times < n_range[1]))
        
        # Find spikes in cluster list
        spike_cluster_mask = np.in1d(clusters, cluster_list)
        
        # Mask
        spike_mask = spike_time_mask & spike_cluster_mask
        sub_spike_times = spike_times[spike_mask]
        sub_clusters = clusters[spike_mask]
        sub_features_masks = features_masks[spike_mask, :, 1]
        
        # Plot each
        for ncol, col in enumerate(got_data.T):
            y_offset = -inter_ch_spacing * ncol
            
            # Plot each spike
            for cluster in cluster_list:
                # Extract stuff just from this cluster
                cluster_mask = sub_clusters == cluster
                cluster_sub_spike_times = sub_spike_times[cluster_mask]
                cluster_sub_features_masks = sub_features_masks[cluster_mask]
                color = cluster2color[cluster]

                # Plot each spike separately
                for spike_time, spike_feature in zip(
                    cluster_sub_spike_times, cluster_sub_features_masks):
                    # Determine which channels to plot
                    reshaped_spike_feature = spike_feature.reshape(-1, 3)
                    
                    # Not sure what threshold to use and whether any given
                    # channel's features are always all true or all false
                    chmask = (reshaped_spike_feature > .5).any(1)
                    
                    # We are currently plotting ch_list[ncol] (datafile order)
                    # So check whether the mask of this channel is true
                    if chmask[ch_list[ncol]]:
                        # Normalize the spike time to the plotting window
                        nspike_time = spike_time - n_range[0]
                        
                        # The indices to plot, downsampled
                        idx0 = int((nspike_time - 15) / downsample)
                        idx1 = int((nspike_time + 15) / downsample)
                        
                        ax.plot(
                            t_ds[idx0:idx1],
                            col[idx0:idx1] + y_offset,
                            color=color,
                            **plot_kwargs
                            )

    # lims
    ax.set_ylabel('channels sorted by depth', size='small')
    ax.set_ylim((-inter_ch_spacing * len(ch_list), 2 * inter_ch_spacing))
    ax.set_xlim((t[0], t[-1]))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_frame_on(False)
    
    ax.plot(
        [t[0] + legend_t_offset, t[0] + legend_t_offset], 
        [legend_y_offset, legend_y_offset + 200], 'k-')
    ax.plot(
        [t[0] + legend_t_offset, t[0] + legend_t_offset + legend_t_width], 
        [legend_y_offset, legend_y_offset], 'k-')
    ax.text(t[0] + legend_t_offset - .002, legend_y_offset + 100, '200 uV', 
        ha='right', va='center', size='small')
    ax.text(t[0] + legend_t_offset + .005, legend_y_offset - 25, 
        '%d ms' % int(legend_t_width * 1000), 
        ha='center', va='top', size='small')
    
    if legend_y_offset + 200 > ax.get_ylim()[1]:
        ax.set_ylim((ax.get_ylim()[0], legend_y_offset + 200))
    
    return got_data

def extract_onsets_from_analog_signal(h5_node, h5_col=35, quick_stride=15000,
    thresh_on=2**14):
    """Find the times that the analog signal went high
    
    First searches coarsely by subsampling by quick_stride
    Then searches more finely around each hit
    """
    # Coarse search
    coarse = h5_node[::quick_stride, h5_col]
    coarse_onsets = np.where(np.diff((coarse > thresh_on).astype(np.int)) == 1)[0]
    # For each coarse_onset, we know that coarse[coarse_onset+1] is >thresh,
    # and coarse[coarse_onset] is <= thresh.

    # Now find the exact onset in the range around coarse onset
    onsets = []
    for coarse_onset in coarse_onsets:
        # Slice around the coarse hit
        # We know it has to be somewhere in
        # [coarse_onset*quick_stride:(coarse_onset+1)*quick_stride], but
        # inclusive of the right limit.
        slc = h5_node[
            coarse_onset*quick_stride:(coarse_onset+1)*quick_stride + 1, 
            h5_col]
        fine_onsets = np.where(np.diff((slc > thresh_on).astype(np.int)) == 1)[0]
        
        if len(fine_onsets) == 0:
            raise ValueError("no onset found, but supposed to be")
        elif len(fine_onsets) > 1:
            raise ValueError("too many onsets in slice")
        
        final_onset = fine_onsets[0] + coarse_onset * quick_stride
        onsets.append(final_onset)
    
    return np.asarray(onsets)

def sync_behavior_and_neural(neural_syncing_signal_filename, behavior_filename):
    """Sync neural and behavior
    
    neural_syncing_signal_filename : filename of channel with house light signal
        This should be LOW during the sync pulse
    
    behavior_filename : logfile of session
        Will use MCwatch.behavior.syncing.get_light_times_from_behavior_file
        to get the light times
    
    Returns: b2n_fit
    """
    # Extract the ain36 signal
    chdata = my.OpenEphys.loadContinuous(neural_syncing_signal_filename,
        dtype=np.int16)
    rawdata = np.transpose([chdata['data']])

    # Convert HIGH to LOW
    rawdata = 2**15 - rawdata

    # The light should always be on for at least 0.5s, prob more
    # So first threshold by striding over 0.5s intervals, then refine
    hlight_col = 0
    n_onsets = extract_onsets_from_analog_signal(rawdata, hlight_col, 
        quick_stride=100) / 30e3

    # Extract light on times from behavior file
    b_light_on, b_light_off = MCwatch.behavior.syncing.get_light_times_from_behavior_file(
        logfile=behavior_filename)
    lines = ArduFSM.TrialSpeak.read_lines_from_file(behavior_filename)
    parsed_df_by_trial = \
        ArduFSM.TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Find the time of transition into inter_trial_interval (13)
    backlight_times = ArduFSM.TrialSpeak.identify_state_change_times(
        parsed_df_by_trial, state1=1, show_warnings=True)

    b2n_fit = MCwatch.behavior.syncing.longest_unique_fit(n_onsets, backlight_times)
    return b2n_fit

def load_all_spikes_and_clusters(kwik_path):
    """Load all spikes and clusters from session.
    
    Returns: dict, with these items:
        'spike_times' : a sorted array of all spike times, in seconds
        'clusters' : the cluster identity of each spike time
        'group2cluster' : dict, with these items:
            'noise': array of clusters belong to noise
            'mua': array of clusters belong to mua
            'good': array of clusters belong to good
            'unsorted': array of clusters belong to unsorted
    
    The MUA and MSUA can easily be extracted:
        mua = spike_times[np.in1d(clusters, group2cluster['mua'])]
        msua = spike_times[np.in1d(clusters, group2cluster['good'])]
    """
    ## Load spikes
    # 0=Noise, 1=MUA, 2=Good, 3=Unsorted
    with tables.open_file(kwik_path, 'r') as h5file:
        # Get all the unique cluster numbers
        clusters = h5file.get_node('/channel_groups/0/spikes/clusters/main')[:]
        unique_clusters = np.unique(clusters)
        
        # Arrange the cluster numbers by the type of cluster (noise, etc)
        group2key = {0: 'noise', 1: 'mua', 2: 'good', 3: 'unsorted'}
        group2cluster = {'noise': [], 'mua': [], 'good': [], 'unsorted': []}
        for cluster in unique_clusters:
            cg = h5file.get_node_attr(
                '/channel_groups/0/clusters/main/%d' % cluster,
                'cluster_group')
            key = group2key[cg]
            group2cluster[key].append(cluster)
        
        # Get all of the spike times
        spike_times = h5file.get_node(
            '/channel_groups/0/spikes/time_samples')[:] / 30e3

    # Sort spike times
    sortmask = np.argsort(spike_times)
    spike_times = spike_times[sortmask]
    clusters = clusters[sortmask]

    return {
        'spike_times': spike_times,
        'clusters': clusters,
        'group2cluster': group2cluster,
    }

def load_features(kwx_path):
    """Loads features from kwx file
    
    Returns: features
        array with shape (n_spikes, n_features)
        n_features = n_channels * 3
    """
    # Load features masks
    # this is n_spikes x n_features x 2
    with tables.open_file(kwx_path, 'r') as h5file:
        features_masks = h5file.get_node('/channel_groups/0/features_masks')
        features = features_masks[:, :, 0]
    return features

def load_masks(kwx_path):
    """Loads masks from kwx file
    
    We subsample the masks by 3 since they are redundant over features
    
    Returns: masks
        array with shape (n_spikes, n_channels)
    """    
    # Load features masks
    # this is n_spikes x n_features x 2
    with tables.open_file(kwx_path, 'r') as h5file:
        features_masks = h5file.get_node('/channel_groups/0/features_masks')
        masks = features_masks[:, ::3, 1]
    return masks


def lock_spikes_to_events(spike_times, event_times, dstart, dstop,
    spike_range_t, event_range_t):
    """Lock spike times to event times and return Folded
    
    spike_times : spike times
    event_times : event times
    dstart, dstop : intervals to pass to Folded
    spike_range_t : earliest and latest possible time of spikes
    event_times_t : earliest and latest possible event times
    
    Only spikes and events in the overlap of the spike and event intervals
    are included. For convenience dstart is added to the start and dstop
    is added to the stop of the overlap interval.
    """
    t_start = np.max([spike_range_t[0], event_range_t[0]]) + dstart
    t_stop = np.min([spike_range_t[1], event_range_t[1]]) + dstop
    spike_times = spike_times[
        (spike_times >= t_start) &
        (spike_times < t_stop)
    ]
    event_times = event_times[
        (event_times >= t_start) &
        (event_times < t_stop)
    ]    
    
    folded = kkpandas.Folded.from_flat(spike_times, centers=event_times,
        dstart=dstart, dstop=dstop)
    
    return folded

def get_channel_mapping(sorted_channels_to_remove):
    """Returns dataflow channel mapping, leaving out certain channels.
    
    sorted_channels_to_remove : list of channels to remove, using the GUI sorted
        numbering. (Same as in probe file naming.)
    
    Returns : dataflow df with channels removed
        Also adds a 'kwx_order' column which is the index into the kwx file,
        which is Intan numbering accounting for missing channels
    """
    # Main adapter dataflow
    dataflow = Adapters.dataflow.dataflow_janelia_64ch_ON2_df

    # Ensure it is sorted by Srt
    dataflow = dataflow.sort_values(by='Srt')

    # Drop the broken channels
    dataflow_minus = dataflow.ix[~dataflow.Srt.isin(
        sorted_channels_to_remove)].copy()
    
    dataflow_minus['kwx_order'] = dataflow_minus['Int'].rank().astype(np.int) - 1
    
    return dataflow_minus