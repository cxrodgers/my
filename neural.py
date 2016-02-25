"""Neural data stuff"""
import OpenEphys
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

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
    cluster2color=None, legend_t_width=.010, apply_offset=None):
    """Plot a vertical stack of channels in the same ax.
    
    The given time range and channel range is extracted. Optionally it
    can be filtered at this point. 
    """
    # Set up the channels to include
    if ch_list is None:
        if probename is None:
            ch_list = list(range(32))
        else:
            ch_list = probename2ch_list[probename]
    if exclude_ch_list is not None:
        for ch in exclude_ch_list:
            ch_list.remove(ch)

    # Set up data_range
    if n_range is None:
        n_range = (0, len(data))
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