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
import os
import pandas
import scipy.io

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

def extract_onsets_from_analog_signal(sync_signal, quick_stride=15000,
    thresh_on=2**14, invert=True):
    """Find the times that the analog signal went high
    
    First searches coarsely by subsampling by quick_stride
    Then searches more finely around each hit
    
    sync_signal : 1d array
    quick_stride : how to coarsely subsample
    thresh_on : threshold crossing
    invert : if True, identify when signal goes low    
        This is done by (2**15 - 1) - signal so assumes integer!
    """
    if invert:
        sync_signal = 2 ** 15 - 1 - sync_signal
    
    # Coarse search
    coarse = sync_signal[::quick_stride]
    coarse_onsets = np.where(
        np.diff((coarse > thresh_on).astype(np.int)) == 1)[0]
    # For each coarse_onset, we know that coarse[coarse_onset+1] is >thresh,
    # and coarse[coarse_onset] is <= thresh.

    # Now find the exact onset in the range around coarse onset
    onsets = []
    for coarse_onset in coarse_onsets:
        # Slice around the coarse hit
        # We know it has to be somewhere in
        # [coarse_onset*quick_stride:(coarse_onset+1)*quick_stride], but
        # inclusive of the right limit.
        slc = sync_signal[
            coarse_onset*quick_stride:(coarse_onset+1)*quick_stride + 1]
        fine_onsets = np.where(
            np.diff((slc > thresh_on).astype(np.int)) == 1)[0]
        
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
    spike_range_t, event_range_t, event_labels=None):
    """Lock spike times to event times and return Folded
    
    spike_times : spike times
    event_times : event times
    dstart, dstop : intervals to pass to Folded
    spike_range_t : earliest and latest possible time of spikes
    event_times_t : earliest and latest possible event times
    event_labels : label of each event
        These will be assigned to folded.labels, after masking them
        in the same way as event_times
    
    Only spikes and events in the overlap of the spike and event intervals
    are included. For convenience dstart is added to the start and dstop
    is added to the stop of the overlap interval.
    """
    # Arrayify
    spike_times = np.asarray(spike_times)
    event_times = np.asarray(event_times)
    if event_labels is not None:
        event_labels = np.asarray(event_labels)
    
    # Error check
    if event_times.ndim != 1:
        raise ValueError("event times must be 1d")
    if spike_times.ndim != 1:
        raise ValueError("spike times must be 1d")    
    
    # Identify temporal range
    t_start = np.max([spike_range_t[0], event_range_t[0]]) + dstart
    t_stop = np.min([spike_range_t[1], event_range_t[1]]) + dstop
    
    # Mask spike times and events
    spike_times = spike_times[
        (spike_times >= t_start) &
        (spike_times < t_stop)
    ]
    event_mask = (
        (event_times >= t_start) & 
        (event_times < t_stop)
    )
    event_times = event_times[event_mask]
    
    # Mask event labels
    if event_labels is not None:
        event_labels = event_labels[event_mask]
    
    # Fold and assign labels
    folded = kkpandas.Folded.from_flat(spike_times, centers=event_times,
        dstart=dstart, dstop=dstop)
    folded.labels = event_labels
    
    return folded

def get_dataflow_accounting_for_missing(sorted_channels_to_remove, 
    probe, adapter='ON4'):
    """Returns dataflow channel mapping, leaving out certain channels.
    
    sorted_channels_to_remove : list of channels to remove, using the GUI 
        sorted numbering.
    probe : 'janelia' or 'H3' (case-insensitive)
    adapter : 'ON4'
    
    Returns : dataflow df with channels removed
        Also adds a 'Srt_wo_broken' column which is just the channels
        in sorted order not including the broken ones. This is the way
        they are indexed by `cluster_channels`.
    """
    # Main adapter dataflow
    if adapter == 'ON4':
        if probe.lower() == 'janelia':
            dataflow = Adapters.dataflow.dataflow_janelia_64ch_ON4_df
        elif probe.lower() == 'h3':
            dataflow = Adapters.dataflow.dataflow_h3_ON4_df
        else:
            raise ValueError("probe %s not supported" % probe)
    else:
        raise ValueError("adapter %s not supported" % adapter)

    # Ensure it is sorted by Srt
    dataflow = dataflow.sort_values(by='Srt')

    # Drop the broken channels
    dataflow_minus = dataflow.ix[~dataflow.Srt.isin(
        sorted_channels_to_remove)].copy()
    
    # Add Srt_wo_broken
    dataflow_minus['Srt_wo_broken'] = list(range(len(dataflow_minus)))
    
    return dataflow_minus


## For loading from kilosort
def load_spike_clusters(sort_dir):
    """Load the cluster of each spike from kilosort data
    
    This includes any reclustering that was done in phy
    """
    spike_cluster = np.load(os.path.join(sort_dir, 'spike_clusters.npy'))
    return spike_cluster

def load_spikes(sort_dir):
    """Load spike times from kilosort
    
    Returns: 
        spike_time_samples, spike_time_sec
    """
    spike_time_samples = np.load(
        os.path.join(sort_dir, 'spike_times.npy')).flatten()
    spike_time_sec = spike_time_samples / 30e3
    
    return spike_time_samples, spike_time_sec

def load_spike_templates1(sort_dir):
    """Return spike templates from kilosort

    These are the actual templates that were used, not the templates
    for each spike. For that, use load_spike_templates2
    
    Returns: templates
        array with shape (n_templates, n_timepoints, n_channels)
    """
    templates = np.load(os.path.join(sort_dir, 'templates.npy'))
    return templates

def load_unwhitened_and_scaled_spike_templates(sort_dir):
    """Return the unwhitened spike templates scaled by their amplitudes.
    
    This is the `Wraw` variable in `rez.mat`.
    It has shape (n_templates, n_samples, n_channels)
    https://github.com/cortex-lab/KiloSort/issues/35
    """
    with tables.open_file(os.path.join(sort_dir, 'rez.mat')) as h5_file:
        Wraw = np.asarray(h5_file.get_node('/rez/Wraw'))[:]
    
    return Wraw

def load_spike_amplitudes(sort_dir):
    """Return spike amplitudes from kilosort
    
    """
    # Amplitude of every spike
    spike_amplitude = np.load(os.path.join(sort_dir, 'amplitudes.npy'))
    
    return spike_amplitude.flatten()

def load_spike_templates2(sort_dir):
    """Return template of each spike from kilosort

    They are returned in 0-based indices into `templates`, which can be
    read from load_spike_templates1
    """
    # This is 4 x n_spikes
    # The rows are: spike time, spike template, amplitude, ?? (maybe batch)
    with tables.open_file(os.path.join(sort_dir, 'rez.mat')) as h5_file:
        st3 = np.asarray(h5_file.get_node('/rez/st3'))
    # These are 1-based, so subtract 1
    spike_template = st3[1].astype(np.int) - 1
    #~ assert (st3[0].astype(np.int) == spike_time_samples).all()
    #~ assert (st3[2] == spike_amplitude).all()
    
    return spike_template

def load_cluster_groups(sort_dir):
    """Returns type (good, MUA, noise) of each cluster from kilosort"""
    # This has n_manual_clusters rows, with the group for each
    cluster_group = pandas.read_table(os.path.join(sort_dir, 
        'cluster_group.tsv'))
    
    return cluster_group

def get_n_spikes_by_cluster_and_template(spike_cluster, spike_template):
    """Return counts of how many of each template occur in each cluster"""
    # Identify which templates belong to which clusters
    unique_clusters = np.unique(spike_cluster)
    rec_l = []
    for cluster in unique_clusters:
        msk = spike_cluster == cluster
        cluster_spike_template = spike_template[msk]
        spikes_per_matching_template = pandas.value_counts(cluster_spike_template,
            sort=True)
        rec_l.append(spikes_per_matching_template)
    n_spikes_by_cluster_and_template = pandas.concat(rec_l, keys=unique_clusters)
    n_spikes_by_cluster_and_template.index.names = ('cluster', 'template')
    
    return n_spikes_by_cluster_and_template

def get_cluster_channels(sort_dir, cluster_group, spike_cluster, 
    spike_template, templates):
    """Identify the channel of max power for each cluster's template
    
    sort_dir : path to data
    spike_cluster : cluster of each spike
    spike_template : template of each spike
    templates : the templates
    
    First, for every template, the channel with maximum standard deviation
    is identified. Then, for each cluster (unique value in spike_cluster),
    all matching templates are found. This accounts for any reclustering 
    that was done manually. Finally, for each cluster, a weighted average
    of the channel corresponding to each template (weighted by the prevalence
    of that template in that cluster) is taken.
    
    Channels are just 0-based indices into the templates, so any broken 
    channels have been ignored already.
    
    Returns : cluster_channels
        Series indexed by cluster id with values corresponding to channel
        The values can be float because they are averages over templates.
    """
    n_spikes_by_cluster_and_template = get_n_spikes_by_cluster_and_template(
        spike_cluster, spike_template)
    
    # Identify channel with max power for each template
    template_channel = templates.std(axis=1).argmax(axis=1)
    cluster_channel_l = []
    for cluster in n_spikes_by_cluster_and_template.index.levels[0]:
        n_spikes_by_template = n_spikes_by_cluster_and_template.loc[cluster]
        
        # Weighted average of template_channel by n_spikes_by_template
        cluster_channel = (
            (n_spikes_by_template / n_spikes_by_template.sum()) *
            template_channel[n_spikes_by_template.index.values]
        ).sum()
        cluster_channel_l.append(cluster_channel)
    cluster_channels = pandas.Series(cluster_channel_l, 
        index=n_spikes_by_cluster_and_template.index.levels[0])
    
    return cluster_channels

def extract_peak_and_width(waveform):
    """Return properties of the waveform peak
    
    Typically this is used to identify narrow-spiking cells. I find that
    a good criterion is the width from the peak to the subsequent zero crossing.
    When this value is < 8 samples (0.2667ms), call it narrow-spiking.
    
    waveform : array
    
    Returns : dict of properties
        idx : index of peak (where waveform reaches its greatest absolute value)
        negative : True if that peak is negative
        height : height of the peak (may be negative)
        stop : index where peak crosses zero again after peak
            If it never crosses zero, this will be len(waveform)
        width : difference between `stop` and `idx`
            This is minimal when the waveform crosses zero on the first
            sample after the peak, in which case the `width` is 1
    """
    # Identify polarity and peak
    peak_loc = np.argmax(np.abs(waveform))
    peak_ampl = waveform[peak_loc]
    peak_is_negative = peak_ampl < 0
    
    # Make it always positive polarity for this purpose
    if peak_is_negative:
        pos_waveform = -waveform.copy()
    else:
        pos_waveform = waveform.copy()
    
    # First point that crosses zero after the peak
    mask = (
        (pos_waveform <= 0) &
        (range(len(pos_waveform)) > peak_loc))
    if np.all(~mask):
        after_loc = len(pos_waveform)
    else:
        after_loc = np.where(mask)[0][0]
    
    # The width is the range from peak to zero crossing
    # Minimum possible value is 1
    peak_width = after_loc - peak_loc

    return {
        'idx': peak_loc, 'height': peak_ampl, 'negative': peak_is_negative,
        'stop': after_loc, 'width': peak_width
    }

def calculate_peak_properties(spike_cluster, spike_template, templates):
    """Calculate properties of peak for all clusters
    
    Extracts the average template for each cluster, weighted by the
    occurrence of each template. Then calculates various properties
    of the peak such as width.
    """
    n_spikes_by_cluster_and_template = get_n_spikes_by_cluster_and_template(
        spike_cluster, spike_template)    
    
    # Average the templates by cluster, weighted by number of spikes
    peak_properties_l = []
    for cluster in n_spikes_by_cluster_and_template.index.levels[0]:
        # Templates and number of spikes for this cluster
        n_spikes_by_template = n_spikes_by_cluster_and_template.loc[cluster]
        
        # Extract relevant templates
        cluster_templates = templates[n_spikes_by_template.index, :, :]
        
        # Mean the templates
        mean_cluster_templates = cluster_templates.mean(axis=0)
        
        # Identify channel with max std
        std_mct = mean_cluster_templates.std(axis=0)
        big_ichannel = std_mct.argmax()
        big_waveform = mean_cluster_templates[:, big_ichannel]
        
        # Identify peak
        peak_properties = extract_peak_and_width(big_waveform)
        peak_properties['waveform'] = big_waveform
        
        # Store width
        peak_properties_l.append(peak_properties)
    peak_properties_df = pandas.DataFrame.from_records(peak_properties_l,
        index=n_spikes_by_cluster_and_template.index.levels[0])
    
    return peak_properties_df

def generate_kilosort_channel_map_file(dataflow, 
    output_filename='chmap.mat', exclude_channels=None):
    """Generate the channel map .mat file expected by KiloSort
    
    dataflow : eg Adapters.dataflow.dataflow_janelia_64ch_ON4_df
    output_filename : filename
    exclude_channels : the 1-based GUI numbers of channels to exclude
    """
    # Normally included channels
    gui_channels = dataflow['GUI'].values

    # Specify the channels to exclude, in GUI numbers
    if exclude_channels is None:
        exclude_channels_zerobased = np.asarray([], dtype=np.int)
    else:
        exclude_channels_zerobased = np.asarray(exclude_channels).astype(
            np.int) - 1

    # Mask
    # To exclude Sorted27, make the 27th entry false
    # Not the location of 27 in the GUI numbers
    n_channels = len(dataflow)
    connected_mask = np.ones(n_channels).astype(np.bool)
    connected_mask[exclude_channels_zerobased] = False

    data = {
        'chanMap': gui_channels,
        'chanMap0ind': gui_channels - 1,
        'connected': connected_mask,
        'xcoords': np.ones(n_channels),
        'ycoords': np.arange(n_channels, dtype=np.float)[::-1] + 1,
        'kcoords': np.ones(n_channels),
        'fs': 30000.0,
    }

    scipy.io.savemat('chmap.mat', data, oned_as='column')

def infer_epochs_and_identify_munged_records(timestamps, error_thresh=30e3,
    mung_mask_size=10):
    """Use timestamps jumps to identify epochs and errors
    
    timestamps : the timestamps from any OpenEphys channel
        I think this should be the same for all channels?
    
    Note: seems like the epoch number is already encoded as "recordingNumber",
    though need to divide by 256.
    
    Normally, each timestamp should be separated by 1024 samples. Starting 
    and stopping the recording causes a jump. Occasional errors also cause 
    jumps. Jumps less than `error_thresh` samples are assumed to be an error.
    
    Disregarding error jumps, the "epoch" begins at 0 and increments at
    every jump.
    
    `munged_mask` is a boolean array of the same shape as `timestamps` and
    is True whenever that timestamp is within `mung_mask_size` records of
    an erroneous jump.
    
    To identify epochs of neural data to discard, it's probably best to 
    convert `error_jump_record_indices` to samples by multiplying by 1024,
    and using `times_near_times` logic similar to here.
    
    To "fix" spike times by accounting for jumps, convert them to records
    by dividing by 1024 and flooring, then index into timestamps, then add
    that to the original spike time mod 1024.
    
    Returns: dict
        'error_jump_record_indices': indices of the records following
            erroneous jumps
        'munged_mask': see above
        'epoch_of_each_record': integer array, same shape as timestamps, the
            inferred epoch of each timestamp. This always begins with 0.
        'epoch_start_record': integer array of the start of each record.
            This always begins with 0.
    
    For instance, if there is a real jump between the first and second
    timestamps, 'epoch_of_each_record' will be [0, 1, ...] and
    'epoch_start_record' will be [0, 1, ...]

    Example with a real jump
    ---
    In [38]: timestamps = np.array([0, 1024, 2048, 100000, 101024])

    In [39]: my.neural.infer_epochs_and_identify_munged_records(timestamps)
    Out[39]: 
    {'epoch_of_each_record': array([0, 0, 0, 1, 1]),
     'epoch_start_record': array([0, 3]),
     'error_jump_record_indices': array([], dtype=int64),
     'munged_mask': array([False, False, False, False, False, False], dtype=bool)}


    Example with an erroneous jump
    ---
    In [40]: timestamps = np.array([0, 1024, 2048, 10000, 11024])

    In [42]: my.neural.infer_epochs_and_identify_munged_records(timestamps, 
                mung_mask_size=1)
    Out[42]: 
    {'epoch_of_each_record': array([0, 0, 0, 0, 0]),
     'epoch_start_record': array([0]),
     'error_jump_record_indices': array([2]),
     'munged_mask': array([False, False,  True,  True, False, False], dtype=bool)}
    """
    ## Infer epochs and identify munged periods
    # Each timestamp *should* be 1024 samples apart
    diff_timestamps = np.diff(timestamps)
    
    # Failures manifest as jumps that are >1024 but <30e3 (1 sec)
    putative_failure_mask = (diff_timestamps > 1024) & (
        diff_timestamps < error_thresh)
    error_jump_record_indices = np.where(putative_failure_mask)[0]
    
    # Mask out records within 10 records of a failure
    munged_mask = my.misc.times_near_times(error_jump_record_indices,
        np.arange(len(timestamps), dtype=np.int), 
        dstart=-mung_mask_size, dstop=mung_mask_size)
    
    # Account for the diff
    munged_mask = np.concatenate([[munged_mask[0]], munged_mask])
    
    # Identify epochs as jumps that are >30e3 (1sec)
    epoch_start_record_mask = ((diff_timestamps > 1024) & 
        (~putative_failure_mask))
    epoch_start_record = np.where(epoch_start_record_mask)[0]
    epoch_of_each_record = epoch_start_record_mask.cumsum()
    
    # Account for the diff. We always start with epoch 0
    epoch_start_record = np.concatenate([[0], epoch_start_record + 1]).astype(
        np.int)
    epoch_of_each_record = np.concatenate([[0], epoch_of_each_record]).astype(
        np.int)

    return {
        'error_jump_record_indices': error_jump_record_indices,
        'munged_mask': munged_mask,
        'epoch_of_each_record': epoch_of_each_record,
        'epoch_start_record': epoch_start_record,
    }
