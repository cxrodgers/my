import os
import my
import glob
import pandas
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import my.plot

def parse_tiffs_into_array(session_root_path):
    """For each subdirectory, load all tiffs and parase into array.
    
    Saves the final array into session_root_path using the filename:
        image_data_SESSION_NAME.npy
    """
    session_dirs = sorted(glob.glob(os.path.join(session_root_path, '*/')))

    for session_dir in session_dirs:
        _parse_tiffs_into_array(session_dir, session_root_path)

def _parse_tiffs_into_array(session_dir, session_root_path):
    session_name = os.path.split(os.path.normpath(session_dir))[1]

    file_list = sorted(os.listdir(session_dir))

    pat = 'trial(\d+) frame(\d+)\.tif'
    file_list = my.misc.regex_filter(pat, file_list)
    trial_nums = my.misc.regex_capture(pat, file_list, 0)
    frame_nums = my.misc.regex_capture(pat, file_list, 1)
    file_df = pandas.DataFrame(
        {'filename': file_list, 'trial': map(int, trial_nums),
            'frame': map(int, frame_nums)})

    trial_list = []
    for trial in file_df.trial.unique():
        print "loading trial", trial
        frame_list = []
        for frame in file_df.frame.unique():
            filename_rows = file_df[
                (file_df.trial == trial) & (file_df.frame == frame)]
            assert len(filename_rows) == 1
            filename = filename_rows.iloc[0]['filename']
            
            img = plt.imread(os.path.join(session_dir, filename))
            frame_list.append(img)
        trial_list.append(frame_list)

    # ntrials, nframes, nrows (?), ncols
    print "saving session", session_name
    arr = np.asarray(trial_list)
    np.save(
        os.path.join(session_root_path, 'image_data_%s' % session_name), 
        arr)

## various plotting functions
def make_slideshow(image3d, c_panels=10, r_panels=None):
    """Panelized a 3d image"""
    n_panels, n_rows, n_cols = image3d.shape
    if r_panels is None:
        r_panels = n_panels / c_panels

    # Concatenate into a big row of panels
    concatted = np.concatenate(image3d, axis=1)
    
    # Make separate rows of panels
    panel_l = [concatted[:, n*n_cols:(n+c_panels)*n_cols] for n in range(5)]
    
    # Concatenate rows of panels
    return np.concatenate(panel_l)

def plot_panels(image3d):
    f, axa = my.plot.auto_subplot(image3d.shape[0])
    for frame_arr, ax in zip(image3d, axa.flatten()):
        my.plot.imshow(frame_arr, ax=ax, cmap=plt.cm.hot, axis_call='image',
            interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
    my.plot.harmonize_clim_in_subplots(fig=f, trim=.95)    
    return f


def process_data_into_effect(session_root_path, REBIN_FACTOR=16,
    baseline_start=0, baseline_stop=10,
    stim_start=16, stim_stop=36,
    session2include_trials=None,
    plot_timecourse=False,
    ):
    """Go through saved numpy arrays, process, save 'effect'
    
    For each image_data* filename in the root path:
        Load it
        Rebin it by REBIN_FACTOR
        Average over baselined and stim periods for each trial
        Subtract baseline from stim for each trial
        Mean the result over trials
    
    The start and stop frames are Pythonic (half-open)
    
    
    session2include_trials : if not None, should be a dict from
        session to a list of trials to include
    
    plot_timecourse: plot time course over average trial
    
    Notes:
    The dominating effect is a decrease in luminance over the session.
    Within each trial, there is sometimes a large blood vessel artefact in 
    either the baseline or the stim period or both.
    This artefact doesn't usually subtract out. Taking the median within each
    time period doesn't help either. So, probably the only way to get rid of
    it would be to decompose the image into two factors, which is too complicated.
    Or, identify trials with significantly higher variance and drop them
 
    In choosing the baseline and stim periods, balance longer periods for
    more signal, vs shorter periods for less likelihood of vessel shift
    
    The time course of the evoked effect seems highly variable across sessions,
    sometimes early, sometimes late, sometimes broadening with time but not
    always. In general, it's probably best to use large baseline windows
    (first ten frames or so) and large evoked windows (frames 16-36 or even
    longer)
    
    
    Returns: 
        session2effect
            dict with sessions as keys, and effects as values
        session2effect_by_trial
            dict with sessions as keys, and 3d effects as values
    """
    session2effect = {}
    session2effect_by_trial = {}

    # Find all image_data* filenames and extract session names
    session_filenames = sorted(glob.glob(
        os.path.join(session_root_path, 'image_data*')))
    session_names = []
    for session_filename in session_filenames:
        session_short_filename = os.path.split(
            os.path.normpath(session_filename))[1]
        session_name = os.path.splitext(session_short_filename)[0].split('_')[-1]
        session_names.append(session_name)

    # Process each
    for session_name, session_filename in zip(session_names, session_filenames):
        print "processing session", session_name
        
        # Load data
        image_data = np.load(session_filename)
        n_trials, n_frames, n_rows, n_cols = image_data.shape
        
        # Slice out trials
        if session2include_trials is not None:
            if session_name in session2include_trials:
                include_trials = session2include_trials[session_name]
                image_data = image_data[include_trials, :, :]
                assert len(include_trials) == image_data.shape[0]
                n_trials = image_data.shape[0]
        
        # Rebin and convert to float
        print "rebinning"
        rebinned_image_data = image_data.reshape(
            (n_trials, n_frames, 
                n_rows / REBIN_FACTOR, REBIN_FACTOR,
                n_cols / REBIN_FACTOR, REBIN_FACTOR)).mean(axis=(-3, -1))

        if plot_timecourse:
            # Display the trial average normalized to the mean of the first ten
            trial_average = rebinned_image_data.mean(axis=0)
            baseline_trial_average = trial_average[5:10].mean(0)
            trial_average_wrt_baseline = trial_average - baseline_trial_average
            
            # this looks ugly for some reason
            #~ ax = my.plot.imshow(make_slideshow(trial_average_wrt_baseline), cmap=plt.cm.hot)

            f = plot_panels(trial_average_wrt_baseline)
            f.suptitle('trial average')

        # Compute average over all trials
        print "baselining"
        
        # Compute the mean baseline and mean stim for each trial separately
        baseline_period_by_trial = rebinned_image_data[:, 
            baseline_start:baseline_stop].mean(1)
        stim_period_by_trial = rebinned_image_data[:, 
            stim_start:stim_stop].mean(1)
        effect_by_trial = stim_period_by_trial - baseline_period_by_trial
        
        # For each trial, plot: 
        # baseline vs mean baseline
        # stim vs mean stim
        # (stim-baseline) vs mean (stim-baseline)
        #~ f = plot_panels(baseline_period_by_trial - baseline_period_by_trial.mean(0))
        #~ f.suptitle('baseline by trial wrt mean baseline')
        #~ f = plot_panels(stim_period_by_trial - stim_period_by_trial.mean(0))
        #~ f.suptitle('stim by trial wrt mean baseline')
        #~ f = plot_panels(effect_by_trial - effect_by_trial.mean(0))
        #~ f.suptitle('effect by trial wrt mean effect')

        # Plot mean effect over all trials
        #~ ax = my.plot.imshow(effect_by_trial.mean(0),
            #~ axis_call='image', cmap=plt.cm.hot)
        #~ ax.set_title = 'mean effect'
        
        effect = effect_by_trial.mean(0)

        session2effect[session_name] = effect
        session2effect_by_trial[session_name] = effect_by_trial

    return session2effect, session2effect_by_trial

def put_scale_bars_on_axis(ax, scbar_center=(150, 150), scbar_length_um=100):
    """Puts P-L scale bars on. 
    
    Assumes image has already been rotated to have posterior toward right.
    Assumes the data range has been set in microns    
    """
    # Put scale bars
    scbar_center = np.asarray(scbar_center)

    ax.set_autoscale_on(False)

    # Note the "y-coordinates" are the row indices (the first axis)
    xcoords = np.array([scbar_center[1], scbar_center[1]])
    ycoords = np.array([scbar_center[0], 
        scbar_center[0] + scbar_length_um])
    ax.plot(xcoords, ycoords, color='blue', ls='-')
    ax.text(xcoords.mean(), ycoords[1], 'L',
        color='blue', ha='center', va='top')
    
    xcoords = np.array([scbar_center[0], 
        scbar_center[0] + scbar_length_um])
    ycoords = np.array([scbar_center[1], scbar_center[1]])
    ax.plot(xcoords, ycoords, color='purple', ls='-')
    ax.text(xcoords[1], ycoords.mean(), 'P',
        color='purple', ha='left', va='center')


def plot_rotated_and_scaled_image(image, ax, scale_um_per_px=3.28,
    rotation_angle_deg=-209.2, rebin_factor=1,
    cmap=plt.cm.hot, put_scale_bars=True, cval=None):
    """Rotate and scale image according to intrinsic imaging params.
    
    If the data has been rebinned (e.g., effect) include this factor
    """
    # Rotate the green and the effect
    if cval is None:
        cval = image.mean()
    rot_image = scipy.ndimage.rotate(image, angle=rotation_angle_deg,
        cval=cval)
    
    # Determine the "data range" in microns
    xd_range = (0, rot_image.shape[1] * scale_um_per_px * rebin_factor / 1000.)
    yd_range = (0, rot_image.shape[0] * scale_um_per_px * rebin_factor / 1000.)

    # Plot
    my.plot.imshow(rot_image, ax=ax, 
        cmap=cmap, axis_call='image',
        xd_range=xd_range, yd_range=yd_range,
    )
    
    # Grid every 500um
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.500))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(.500))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))

    if put_scale_bars:
        put_scale_bars_on_axis(ax)
    
    if cmap == plt.cm.gray:
        ax.grid(color='r', ls='-', which='both')    
    else:
        ax.grid(color='w', ls='-', which='both')    

