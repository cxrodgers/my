"""Module for parsing behavior files and video.

These are mainly for dealing with the idiosyncracies of rigs L1,
L2, and L3 in the bruno lab.
"""
import os, numpy as np, glob, re, pandas, datetime
import misc
import subprocess # for ffprobe
import ArduFSM
import scipy.misc

# Known mice
mice = ['AM03', 'AM05', 'KF13', 'KM14', 'KF16', 'KF17', 'KF18', 'KF19', 'KM24', 'KM25']
rigs = ['L1', 'L2', 'L3']
aliases = {
    'KF13A': 'KF13',
    'AM03A': 'AM03',
    }
assert np.all([alias_val in mice for alias_val in aliases.values()])


def load_frames_by_trial(frame_dir, trials_info):
    """Read all trial%03d.png in frame_dir and return as dict"""
    trialnum2frame = {}
    for trialnum in trials_info.index:
        filename = os.path.join(frame_dir, 'trial%03d.png' % trialnum)
        if os.path.exists(filename):
            im = scipy.misc.imread(filename, flatten=True)
            trialnum2frame[trialnum] = im    
    return trialnum2frame

def mean_frames_by_choice(trials_info, trialnum2frame):
    # Keep only those trials that we found images for
    trials_info = trials_info.ix[sorted(trialnum2frame.keys())]

    # Dump those with a spoiled trial
    trials_info = misc.pick_rows(trials_info, choice=[0, 1], bad=False)

    # Split on choice
    res = []
    gobj = trials_info.groupby('choice')
    for choice, subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'choice': choice, 'meaned': meaned})
    resdf_choice = pandas.DataFrame.from_records(res)

    return resdf_choice

def calculate_performance(trials_info, p_servothrow):
    """Use p_servothrow to calculate performance by stim number"""
    rec_l = []
    
    # Assign pos_delta
    raw_servo_positions = np.unique(trials_info.servo_position)
    if len(raw_servo_positions) == 1:
        pos_delta = 25
    else:
        pos_delta = raw_servo_positions[1] - raw_servo_positions[0]
    p_servothrow.pos_delta = pos_delta
    
    # Convert
    ti2 = p_servothrow.assign_trial_type_to_trials_info(trials_info)
    
    # Perf by ST and by SN
    gobj = ti2.groupby(['rewside', 'servo_intpos', 'stim_number'])
    for (rewside, servo_intpos, stim_number), sub_ti in gobj:
        nhits, ntots = ArduFSM.trials_info_tools.calculate_nhit_ntot(sub_ti)
        if ntots > 0:
            rec_l.append({
                'rewside': rewside, 'servo_intpos': servo_intpos,
                'stim_number': stim_number,
                'perf': nhits / float(ntots),
                })

    # Form dataframe
    df = pandas.DataFrame.from_records(rec_l)
    return df



def plot_side_perf(ax, perf):
    """Plot performance on each side vs servo position"""
    colors = ['b', 'r']
    for rewside in [0, 1]:
        # Form 2d perf matrix for this side by unstacking
        sideperf = perf[rewside].unstack() # servo on rows, stimnum on cols
        yvals = map(int, sideperf.index)
        
        # Mean over stim numbers
        meaned_sideperf = sideperf.mean(axis=0)
        
        # Plot
        ax.plot(yvals, sideperf.mean(axis=1), color=colors[rewside])
    
    # Avg over sides
    meaned = perf.unstack(1).mean()
    ax.plot(yvals, meaned, color='k')
    
    ax.set_xlabel('servo position')
    ax.set_ylim((0, 1))
    ax.set_yticks((0, .5, 1))
    ax.set_xticks(yvals) # because on rows
    
    ax.plot(ax.get_xlim(), [.5, .5], 'k:')


def make_overlay(sess_meaned_frames, ax, meth='all'):
    import my.plot
    
    # Split into L and R
    if meth == 'all':
        L = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 0], axis=0)
        R = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 1], axis=0)
    elif meth == 'L':
        closest_L = my.pick_rows(sess_meaned_frames, rewside=0)[
            'servo_pos'].min()
        furthest_R = my.pick_rows(sess_meaned_frames, rewside=1)[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside=0, 
            servo_pos=closest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside=1, 
            servo_pos=furthest_R).irow(0)['meaned']
    elif meth == 'R':
        closest_R = my.pick_rows(sess_meaned_frames, rewside=1)[
            'servo_pos'].min()
        furthest_L = my.pick_rows(sess_meaned_frames, rewside=0)[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside=0, 
            servo_pos=furthest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside=1, 
            servo_pos=closest_R).irow(0)['meaned']     

    # Color them into the R and G space, with zeros for B
    C = np.array([L, R, np.zeros_like(L)])
    C = C.swapaxes(0, 2).swapaxes(0, 1) / 255.

    my.plot.imshow(C, ax=ax, axis_call='image', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])



def cached_dump_frames_at_retraction_times(rows, frame_dir='./frames'):
    """Wrapper around dump_frames_at_retraction_time
    
    Repeats call for each row in rows, as long as the subdir doesn't exist.
    """
    if not os.path.exists(frame_dir):
        print "auto-creating", frame_dir
        os.mkdir(frame_dir)

    # Iterate over sessions
    for idx in rows.index:
        # Something very strange here where iterrows distorts the dtype
        # of the object arrays
        row = rows.ix[idx]

        # Set up output_dir and continue if already exists
        output_dir = os.path.join(frame_dir, row['behave_filename'])
        if os.path.exists(output_dir):
            continue
        else:
            print "auto-creating", output_dir
            os.mkdir(output_dir)
            print output_dir

        # Dump the frames
        dump_frames_at_retraction_time(row, session_dir=output_dir)

def generate_meaned_frames(rows, frame_dir='./frames'):
    """Iterate over rows and extract frames meaned by type.
    
    Get session name from each row.
    Skips any where no frame subdirectory is found.
    
    Returns all meaned frames.
    """
    resdf_d = {}
    for idx, sessrow in rows.iterrows():
        # Check that frame_dir exists
        sess_dir = os.path.join(frame_dir, sessrow['behave_filename'])
        if not os.path.exists(sess_dir):
            continue        
        
        # Load trials info
        trials_info = ArduFSM.trials_info_tools.load_trials_info_from_file(
            sessrow['filename'])

        # Load all images
        trialnum2frame = {}
        for trialnum in trials_info.index:
            filename = os.path.join(sess_dir, 'trial%03d.png' % trialnum)
            if os.path.exists(filename):
                im = scipy.misc.imread(filename, flatten=True)
                trialnum2frame[trialnum] = im

        # Keep only those trials that we found images for
        trials_info = trials_info.ix[sorted(trialnum2frame.keys())]

        # Split on side, servo_pos, stim_number
        res = []
        gobj = trials_info.groupby(['rewside', 'servo_position', 'stim_number'])
        for (rewside, servo_pos, stim_number), subti in gobj:
            meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
                axis=0)
            res.append({'rewside': rewside, 'servo_pos': servo_pos, 
                'stim_number': stim_number, 'meaned': meaned})
        resdf = pandas.DataFrame.from_records(res)

        # Store
        resdf_d[sessrow['behave_filename']] = resdf        

    # Store all results
    meaned_frames = pandas.concat(resdf_d, verify_integrity=True)
    return meaned_frames

def timedelta_to_seconds1(val):
    """Often it ends up as a 0d timedelta array.
    
    This especially happens when taking a single row from a df, which becomes
    a series. Then you sometimes cannot divide by np.timedelta64(1, 's')
    or by 1e9
    """
    ite = val.item() # in nanoseconds
    return ite / 1e9

def timedelta_to_seconds2(val):
    """More preferred ... might have been broken in old versions."""
    return val / np.timedelta64(1, 's')


def dump_frames_at_retraction_time(metadata, session_dir):
    """Dump the retraction time frame into a subdirectory.
    
    metadata : row containing behavior info, video info, and fit info    
    """
    # Load trials info
    trials_info = ArduFSM.trials_info_tools.load_trials_info_from_file(
        metadata['filename'])

    # Insert servo retract time
    splines = ArduFSM.trials_info_tools.load_splines_from_file(
        metadata['filename'])
    trials_info['time_retract'] = \
        ArduFSM.trials_info_tools.identify_servo_retract_times(splines)

    # Fit to video times
    fit = metadata['fit0'], metadata['fit1']
    video_times = trials_info['time_retract'].values - \
        timedelta_to_seconds2(metadata['guess_vvsb_start'])
    trials_info['time_retract_vbase'] = np.polyval(fit, video_times)
    
    # Mask out any frametimes that are before or after the video
    duration_s = timedelta_to_seconds2(metadata['duration_video'])
    mask_by_buffer_from_end(trials_info['time_retract_vbase'], 
        end_time=duration_s, buffer=10)
    
    # Dump frames
    frametimes_to_dump = trials_info['time_retract_vbase'].dropna()
    for trialnum, frametime in trials_info['time_retract_vbase'].dropna().iterkv():
        output_filename = os.path.join(session_dir, 'trial%03d.png' % trialnum)
        misc.frame_dump(metadata['filename_video'], frametime, meth='ffmpeg fast',
            output_filename=output_filename)


def generate_mplayer_guesses_and_sync(metadata, 
    user_results=None, guess=(1., 0.), N=4, pre_time=10):
    """Generates best times to check video, and potentially also syncs.
    
    metadata : a row from bv_files to sync
    
    N times to check in the video are printed out. Typically this is run twice,
    once before checking, then check, then run again now specifying the 
    video times in `user_results`.

    If the initial guess is very wrong, you may need to find a large
    gap in the video and match it up to trials info manually, and use this
    to fix `guess` to be closer.
    """
    # Load trials info
    trials_info = ArduFSM.trials_info_tools.load_trials_info_from_file(
        metadata['filename'])
    splines = ArduFSM.trials_info_tools.load_splines_from_file(
        metadata['filename'])

    # Insert servo retract time
    trials_info['time_retract'] = \
        ArduFSM.trials_info_tools.identify_servo_retract_times(splines)

    # Apply the delta-time guess to the retraction times
    test_guess_vvsb = metadata['guess_vvsb_start'] / np.timedelta64(1, 's')
    trials_info['time_retract_vbase'] = \
        trials_info['time_retract'] - test_guess_vvsb

    # Apply the initial guess on top
    initial_guess = np.asarray(guess)
    trials_info['time_retract_vbase2'] = np.polyval(initial_guess, 
        trials_info['time_retract_vbase'])

    # Choose test times for user
    video_duration = metadata['duration_video'] / np.timedelta64(1, 's')
    test_times, test_next_times = generate_test_times_for_user(
        trials_info['time_retract_vbase'], video_duration,
        initial_guess=initial_guess, N=N)

    # Print mplayer commands
    for test_time, test_next_time in zip(test_times, test_next_times):
        pre_test_time = int(test_time) - pre_time
        print 'mplayer -ss %d %s # guess %0.1f, next %0.1f' % (pre_test_time, 
            metadata['filename_video'], test_time, test_next_time)

    # If no data provided, just return
    if user_results is None:
        return
    if len(user_results) != N:
        print "warning: len(user_results) should be %d not %d" % (
            N, len(user_results))
        return
    
    # Otherwise, fit a correction to the original guess
    new_fit = np.polyfit(test_times.values, user_results, deg=1)
    resids = np.polyval(new_fit, test_times.values) - user_results

    # Composite the two fits
    # For some reason this is not transitive! This one appears correct.
    combined_fit = np.polyval(np.poly1d(new_fit), np.poly1d(initial_guess))

    # Diagnostics
    print os.path.split(metadata['filename'])[-1]
    print os.path.split(metadata['filename_video'])[-1]
    print "combined_fit: %r" % np.asarray(combined_fit)
    print "resids: %r" % np.asarray(resids)    

def search_for_behavior_and_video_files(
    behavior_dir='~/mnt/behave/runmice',
    video_dir='~/mnt/bruno-nix/compressed_eye',
    cached_video_files_df=None,
    ):
    """Get a list of behavior and video files, with metadata.
    
    Looks for all behavior directories in behavior_dir/rignumber.
    Looks for all video files in video_dir.
    Gets metadata about video files using parse_video_filenames.
    Finds which video file maximally overlaps with which behavior file.
    
    TODO: cache the video file probing, which takes a fair amount of time.
    
    Returns: joined, video_files_df
        joined is a data frame with the following columns:
            u'dir', u'dt_end', u'dt_start', u'duration', u'filename', 
            u'mouse', u'rig', u'best_video_index', u'best_video_overlap', 
            u'dt_end_video', u'dt_start_video', u'duration_video', 
            u'filename_video', u'rig_video'
        video_files_df is basically used only to re-cache
    """
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    video_dir = os.path.expanduser(video_dir)
    
    # Acquire all behavior files in the subdirectories
    all_behavior_files = []
    for subdir in rigs:
        all_behavior_files += glob.glob(os.path.join(
            behavior_dir, subdir, 'ardulines.*'))

    # Parse out metadata for each
    behavior_files_df = parse_behavior_filenames(all_behavior_files, 
        clean=True)

    # Acquire all video files
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if len(video_files) == 0:
        print "warning: no video files found"
    video_files_df = parse_video_filenames(video_files, verbose=True,
        cached_video_files_df=cached_video_files_df)

    # Find behavior files that overlapped with video files
    behavior_files_df['best_video_index'] = -1
    behavior_files_df['best_video_overlap'] = 0.0
    
    # Something is really slow in this loop
    for bidx, brow in behavior_files_df.iterrows():
        # Find the overlap between this behavioral session and video sessions
        # from the same rig
        latest_start = video_files_df[
            video_files_df.rig == brow['rig']]['dt_start'].copy()
        latest_start[latest_start < brow['dt_start']] = brow['dt_start']
            
        earliest_end = video_files_df[
            video_files_df.rig == brow['rig']]['dt_end'].copy()
        earliest_end[earliest_end > brow['dt_end']] = brow['dt_end']
        
        # Find the video with the most overlap
        overlap = (earliest_end - latest_start)
        vidx_max_overlap = overlap.argmax()
        
        # Convert from numpy timedelta64 to a normal number
        max_overlap_sec = overlap.ix[vidx_max_overlap] / np.timedelta64(1, 's')
        
        # Store if it's more than zero
        if max_overlap_sec > 0:
            behavior_files_df['best_video_index'][bidx] = vidx_max_overlap
            behavior_files_df['best_video_overlap'][bidx] = max_overlap_sec

    # Join video info
    joined = behavior_files_df.join(video_files_df, on='best_video_index', 
        rsuffix='_video')    
    
    return joined, video_files_df


def parse_behavior_filenames(all_behavior_files, clean=True):
    """Given list of ardulines files, extract metadata and return as df.
    
    Each filename is matched to a pattern which is used to extract the
    rigname, date, and mouse name. Non-matching filenames are discarded.
    
    clean : if True, also clean up the mousenames by upcasing and applying
        aliases. Finally, drop the ones not in the official list of mice.
    """
    # Extract info from filename
    # directory, rigname, datestring, mouse
    pattern = '(\S+)/(\S+)/ardulines\.(\d+)\.(\S+)'
    rec_l = []
    for filename in all_behavior_files:
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(filename))
        if m is not None:
            dir, rig, date_s, mouse = m.groups()

            # The start time is parsed from the filename
            date = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')
            
            # The end time is parsed from the file timestamp
            behavior_end_time = datetime.datetime.fromtimestamp(
                misc.get_file_time(filename))
            
            # Store
            rec_l.append({'dir': dir, 'rig': rig, 'mouse': mouse,
                'dt_start': date, 'dt_end': behavior_end_time,
                'duration': behavior_end_time - date,
                'filename': filename})
    behavior_files_df = pandas.DataFrame.from_records(rec_l)

    if len(behavior_files_df) == 0:
        print "warning: no behavior files found"

    elif clean:
        # Clean the behavior files by upcasing and applying aliases
        behavior_files_df.mouse = behavior_files_df.mouse.apply(str.upper)
        behavior_files_df.mouse.replace(aliases, inplace=True)

        # Drop any that are not in the list of accepted mouse names
        behavior_files_df = behavior_files_df.ix[behavior_files_df.mouse.isin(mice)]

    return behavior_files_df

def parse_video_filenames(video_filenames, verbose=False, 
    cached_video_files_df=None):
    """Given list of video files, extract metadata and return df.

    For each filename, we extract the date (from the filename) and duration
    (using ffprobe).
    
    If cached_video_files_df is given:
        1) Checks that everything in cached_video_files_df.filename is also in
        video_filenames, else errors (because probably something
        has gone wrong, like the filenames are misformatted).
        2) Skips the probing of any video file already present in 
        cached_video_files_df
        3) Concatenates the new video files info with cached_video_files_df
        and returns.
    
    Returns:
        video_files_df, a DataFrame with the following columns: 
            dt_end dt_start duration filename rig
    """
    # Error check
    if cached_video_files_df is not None and not np.all([f in video_filenames 
        for f in cached_video_files_df.filename]):
        raise ValueError("cached_video_files contains unneeded video files")
    
    # Extract info from filename
    # directory, rigname, datestring, extension
    pattern = '(\S+)/(\S+)\.(\d+)\.(\S+)'
    rec_l = []

    for video_filename in video_filenames:
        if video_filename in cached_video_files_df.filename.values:
            continue
        
        if verbose:
            print video_filename
        
        # Match filename pattern
        m = re.match(pattern, os.path.abspath(video_filename))
        if m is None:
            continue
        dir, rig, date_s, video_ext = m.groups()
        
        # Parse the end time using the datestring
        video_end_time = datetime.datetime.strptime(date_s, '%Y%m%d%H%M%S')

        # Video duration and hence start time
        proc = subprocess.Popen(['ffprobe', video_filename],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        res = proc.communicate()[0]

        # Check if ffprobe failed, probably on a bad file
        if 'Invalid data found when processing input' in res:
            # Just store what we know so far and warn
            rec_l.append({'filename': video_filename, 'rig': rig,
                'dt_end': video_end_time,
                })            
            if verbose:
                print "Invalid data found by ffprobe in %s" % video_filename
            continue

        # Parse out start time
        duration_match = re.search("Duration: (\S+),", res)
        assert duration_match is not None and len(duration_match.groups()) == 1
        video_duration_temp = datetime.datetime.strptime(
            duration_match.groups()[0], '%H:%M:%S.%f')
        video_duration = datetime.timedelta(
            hours=video_duration_temp.hour, 
            minutes=video_duration_temp.minute, 
            seconds=video_duration_temp.second,
            microseconds=video_duration_temp.microsecond)
        video_start_time = video_end_time - video_duration
        
        # Store
        rec_l.append({'filename': video_filename, 'rig': rig,
            'dt_end': video_end_time,
            'duration': video_duration,
            'dt_start': video_start_time,
            })

    resdf = pandas.DataFrame.from_records(rec_l)
    
    # Join with cache, if necessary
    if cached_video_files_df is not None:
        if len(resdf) == 0:
            resdf = cached_video_files_df
        else:
            resdf = pandas.concat([resdf, cached_video_files_df], axis=0, 
                ignore_index=True, verify_integrity=True)
    
    return resdf

def mask_by_buffer_from_end(ser, end_time, buffer=10):
    """Set all values of ser to np.nan that occur within buffer of the ends"""
    ser[ser < buffer] = np.nan
    ser[ser > end_time - buffer] = np.nan

def index_of_biggest_diffs_across_arr(ser, ncuts_total=3):
    """Return indices of biggest diffs in various segments of arr"""
    # Cut the series into equal length segments, not including NaNs
    ser = ser.dropna()
    cuts = [ser.index[len(ser) * ncut / ncuts_total] 
        for ncut in range(ncuts_total)]
    cuts.append(ser.index[-1])

    # Iterate over cuts and choose the index preceding the largest gap in the cut
    res = []
    for ncut in range(len(cuts) - 1):
        subser = ser.ix[cuts[ncut]:cuts[ncut+1]]
        res.append(subser.diff().shift(-1).argmax())
    return np.asarray(res)

def generate_test_times_for_user(times, max_time, initial_guess=(.9991, 7.5), 
    N=3, buffer=30):
    """Figure out the best times for a user to identify in the video
    
    times: Series of times in the initial time base.
    initial_guess: linear poly to apply to times as a first guess
    N: number of desired times, taken equally across video
    
    Returns the best times to check (those just before a large gap),
    in the guessed timebase.
    """
    # Apply the second guess, based on historical bias of above method
    new_values = np.polyval(initial_guess, times)
    times = pandas.Series(new_values, index=times.index)
    
    # Mask trials too close to end
    mask_by_buffer_from_end(times, max_time, buffer=buffer)

    # Identify the best trials to use for manual realignment
    test_idxs = index_of_biggest_diffs_across_arr(
        times, ncuts_total=N)
    test_times = times.ix[test_idxs]
    test_next_times = times.shift(-1).ix[test_idxs]
    
    return test_times, test_next_times