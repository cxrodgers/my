"""Module for parsing behavior files and video.

These are mainly for dealing with the idiosyncracies of rigs L1,
L2, and L3 in the bruno lab.
"""
import os, numpy as np, glob, re, pandas, datetime
import misc
import subprocess # for ffprobe
import ArduFSM
import scipy.misc
import my

import sys
tcv2_path = os.path.expanduser('~/dev/ArduFSM/TwoChoice_v2')
if tcv2_path not in sys.path:
    sys.path.append(tcv2_path)

import TrialMatrix, TrialSpeak


# Known mice
mice = ['AM03', 'AM05', 'KF13', 'KM14', 'KF16', 'KF17', 'KF18', 'KF19', 
    'KM24', 'KM25', 'KF26', 'KF28', 'KF30', 'KF32', 'KF33', 'KF35', 'KF36',
    'KF37']
rigs = ['L1', 'L2', 'L3']
aliases = {
    'KF13A': 'KF13',
    'AM03A': 'AM03',
    }
assert np.all([alias_val in mice for alias_val in aliases.values()])

## database management
import socket
LOCALE = socket.gethostname()
if LOCALE == 'chris-pyramid':
    PATHS = {
        'database_root': '/home/chris/mnt/marvin/dev/behavior_db',
        'behavior_dir': '/home/chris/mnt/marvin/runmice',
        'video_dir': '/home/chris/mnt/marvin/compressed_eye',
        }

elif LOCALE == 'marvin':
    PATHS = {
        'database_root': '/home/mouse/dev/behavior_db',
        'behavior_dir': '/home/mouse/runmice',
        'video_dir': '/home/mouse/compressed_eye',
        }

else:
    raise ValueError("unknown locale %s" % LOCALE)


def daily_update():
    """Update the databases with current behavior and video files
    
    This should be run on marvin locale.
    """
    if LOCALE != 'marvin':
        raise ValueError("this must be run on marvin")
    
    daily_update_behavior()
    daily_update_video()
    daily_update_overlap_behavior_and_video()
    
    # TODO:
    # daily_update_trial_matrix

def daily_update_behavior():
    """Update behavior database"""
    # load
    behavior_files_df = search_for_behavior_files(
        behavior_dir=PATHS['behavior_dir'],
        clean=True)
    
    # store copy for error check
    behavior_files_df_local = behavior_files_df.copy()
    
    # locale-ify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')
    
    # save
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')
    behavior_files_df.to_csv(filename, index=False)
    
    # Test the reading/writing is working
    bdf = get_behavior_df()
    if not (behavior_files_df_local == bdf).all().all():
        raise ValueError("read/write error in behavior database")
    
def daily_update_video():
    """Update video database"""
    # find video files
    video_files = glob.glob(os.path.join(PATHS['video_dir'], '*.mp4'))
    
    # TODO: load existing video files and use as cache
    # TODO: error check here; if no videos; do not trash cache
    
    # Parse into df
    video_files_df = parse_video_filenames(video_files, verbose=False,
        cached_video_files_df=None)

    # store copy for error check
    video_files_df_local = video_files_df.copy()

    # locale-ify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        PATHS['video_dir'], '$video_dir$')
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'video.csv')
    video_files_df.to_csv(filename, index=False)    
    
    # Test the reading/writing is working
    vdf = get_video_df()
    if not (video_files_df_local == vdf).all().all():
        raise ValueError("read/write error in video database")    

def daily_update_overlap_behavior_and_video():
    """Update the linkage betweeen behavior and video df
    
    Should run daily_update_behavior and daily_update_video first
    """
    # Load the databases
    behavior_files_df = get_behavior_df()
    video_files_df = get_video_df()

    # Find the best overlap
    new_behavior_files_df = find_best_overlap_video(
        behavior_files_df, video_files_df)
    
    # Join video info
    joined = new_behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')
    
    # Drop on unmatched
    joined = joined.dropna()
    
    # Add the delta-time guess
    # Negative timedeltas aren't handled by to_timedelta in the loading function
    # So store as seconds here
    guess = joined['dt_start_video'] - joined['dt_start']
    joined['guess_vvsb_start'] = guess / np.timedelta64(1, 's')
    
    # locale-ify
    joined['filename'] = joined['filename'].str.replace(
        PATHS['behavior_dir'], '$behavior_dir$')    
    joined['filename_video'] = joined['filename_video'].str.replace(
        PATHS['video_dir'], '$video_dir$')    
        
    # Save
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    joined.to_csv(filename, index=False)

def daily_update_trial_matrix(start_date=None, verbose=False):
    """Cache the trial matrix for every session
    
    TODO: use cache
    """
    # Get
    behavior_files_df = get_behavior_df()
    
    # Filter by those after start date
    behavior_files_df = behavior_files_df[ 
        behavior_files_df.dt_start >= start_date]
    
    # Calculate trial_matrix for each
    session2trial_matrix = {}
    for irow, row in behavior_files_df.iterrows():
        # Check if it already exists
        filename = os.path.join(PATHS['database_root'], 'trial_matrix', 
            row['session'])
        if os.path.exists(filename):
            continue

        if verbose:
            print filename

        # Otherwise make it
        trial_matrix = TrialMatrix.make_trial_matrix_from_file(row['filename'])
        
        # And store it
        trial_matrix.to_csv(filename)

def daily_update_perf_metrics(start_date=None, verbose=False):
    """Calculate simple perf metrics for anything that needs it.
    
    start_date : if not None, ignores all behavior files before this date
        You can also pass a string like '20150120'
    
    This assumes trial matrices have been cached for all sessions in bdf.
    Should error check for this.
    """
    # Get
    behavior_files_df = get_behavior_df()

    # Filter by those after start date
    behavior_files_df = behavior_files_df[ 
        behavior_files_df.dt_start >= start_date]

    # Load what we've already calculated
    pmdf = get_perf_metrics()

    # Calculate any that need it
    new_pmdf_rows_l = []
    for idx, brow in behavior_files_df.iterrows():
        # Check if it already exists
        session = brow['session']
        if session in pmdf['session'].values:
            if verbose:
                print "skipping", session
            continue
        
        # Otherwise run
        trial_matrix = get_trial_matrix(session)
        metrics = calculate_perf_metrics(trial_matrix)
        
        # Store
        metrics['session'] = session
        new_pmdf_rows_l.append(metrics)
    
    # Join on the existing pmdf
    new_pmdf_rows = pandas.DataFrame.from_records(new_pmdf_rows_l)
    new_pmdf = pandas.concat([pmdf, new_pmdf_rows],
        verify_integrity=True,
        ignore_index=True)
    
    # Columns are sorted after concatting
    # Re-use original, this should be specified somewhere though
    if new_pmdf.shape[1] != pmdf.shape[1]:
        raise ValueError("missing/extra columns in perf metrics")
    new_pmdf = new_pmdf[pmdf.columns]
    
    # Save
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    new_pmdf.to_csv(filename, index=False)

def get_perf_metrics():
    """Return the df of perf metrics over sessions"""
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')

    try:
        pmdf = pandas.read_csv(filename)
    except IOError:
        raise IOError("cannot find perf metrics database at %s" % filename)
    
    return pmdf

def flush_perf_metrics():
    """Create an empty perf metrics file"""
    filename = os.path.join(PATHS['database_root'], 'perf_metrics.csv')
    columns=['session', 'n_trials', 'spoil_frac',
        'perf_all', 'perf_unforced',
        'fev_corr_all', 'fev_corr_unforced',
        'fev_side_all', 'fev_side_unforced',
        'fev_stay_all','fev_stay_unforced',
        ]

    pmdf = pandas.DataFrame(np.zeros((0, len(columns))), columns=columns)
    pmdf.to_csv(filename, index=False)

def get_trial_matrix(session):
    filename = os.path.join(PATHS['database_root'], 'trial_matrix', session)
    res = pandas.read_csv(filename)
    return res

def get_all_trial_matrix():
    all_filenames = glob.glob(os.path.join(
        PATHS['database_root'], 'trial_matrix', '*'))
    
    session2trial_matrix = {}
    for filename in all_filenames:
        session = os.path.split(filename)[1]
        trial_matrix = pandas.read_csv(filename)
        session2trial_matrix[session] = trial_matrix
    
    return session2trial_matrix

def get_behavior_df():
    """Returns the current behavior database"""
    filename = os.path.join(PATHS['database_root'], 'behavior.csv')

    try:
        behavior_files_df = pandas.read_csv(filename, 
            parse_dates=['dt_end', 'dt_start', 'duration'])
    except IOError:
        raise IOError("cannot find behavior database at %s" % filename)
    
    # de-localeify
    behavior_files_df['filename'] = behavior_files_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])
    
    # Alternatively, could store as floating point seconds
    behavior_files_df['duration'] = pandas.to_timedelta(
        behavior_files_df['duration'])
    
    return behavior_files_df
    
def get_video_df():
    """Returns the current video database"""
    filename = os.path.join(PATHS['database_root'], 'video.csv')

    try:
        video_files_df = pandas.read_csv(filename,
            parse_dates=['dt_end', 'dt_start'])
    except IOError:
        raise IOError("cannot find video database at %s" % filename)

    # de-localeify
    video_files_df['filename'] = video_files_df['filename'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    
    # Alternatively, could store as floating point seconds
    video_files_df['duration'] = pandas.to_timedelta(
        video_files_df['duration'])    
    
    return video_files_df

def get_synced_behavior_and_video_df():
    """Return the synced behavior/video database"""
    filename = os.path.join(PATHS['database_root'], 'behave_and_video.csv')
    
    try:
        synced_bv_df = pandas.read_csv(filename, parse_dates=[
            'dt_end', 'dt_start', 'dt_end_video', 'dt_start_video'])
    except IOError:
        raise IOError("cannot find synced database at %s" % filename)
    
    # Alternatively, could store as floating point seconds
    synced_bv_df['duration'] = pandas.to_timedelta(
        synced_bv_df['duration'])    
    synced_bv_df['duration_video'] = pandas.to_timedelta(
        synced_bv_df['duration_video'])    

    # de-localeify
    synced_bv_df['filename_video'] = synced_bv_df['filename_video'].str.replace(
        '\$video_dir\$', PATHS['video_dir'])
    synced_bv_df['filename'] = synced_bv_df['filename'].str.replace(
        '\$behavior_dir\$', PATHS['behavior_dir'])        
    
    return synced_bv_df    

def get_manual_sync_df():
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    
    try:
        manual_bv_sync = pandas.read_csv(filename).set_index('session')
    except IOError:
        raise IOError("cannot find manual sync database at %s" % filename)    
    
    return manual_bv_sync

def set_manual_bv_sync(session, sync_poly):
    """Store the manual behavior-video sync for session"""
    
    # Load any existing manual results
    manual_sync_df = get_manual_sync_df()
    
    sync_poly = np.asarray(sync_poly) # indexing is backwards for poly
    
    # Add
    if session in manual_sync_df.index:
        raise ValueError("sync already exists for %s" % session)
    
    manual_sync_df = manual_sync_df.append(
        pandas.DataFrame([[sync_poly[0], sync_poly[1]]],
            index=[session],
            columns=['fit0', 'fit1']))
    manual_sync_df.index.name = 'session' # it forgets
    
    # Store
    filename = os.path.join(PATHS['database_root'], 'manual_bv_sync.csv')
    manual_sync_df.to_csv(filename)

def interactive_bv_sync():
    """Interactively sync behavior and video"""
    # Load synced data
    sbvdf = get_synced_behavior_and_video_df()

    # TODO: join on manual results here

    # Choose session
    choices = sbvdf[['session', 'mouse', 'dt_start', 'best_video_overlap', 'rig']]
    print "Here are the most recent sessions:"
    print choices[-20:]
    choice = None
    while choice is None:
        choice = raw_input('Which index to analyze? ')
        try:
            choice = int(choice)
        except ValueError:
            pass
    test_row = sbvdf.ix[choice]

    # Run sync
    N_pts = 3
    sync_res0 = generate_mplayer_guesses_and_sync(test_row, N=N_pts)

    # Get results
    n_results = []
    for n in range(N_pts):
        res = raw_input('Enter result: ')
        n_results.append(float(res))

    # Run sync again
    sync_res1 = generate_mplayer_guesses_and_sync(test_row, N=N_pts,
        user_results=n_results)

    # Store
    res = raw_input('Confirm insertion [y/N]? ')
    if res == 'y':
        set_manual_bv_sync(test_row['session'], 
            sync_res1['combined_fit'])
        print "inserted"
    else:
        print "not inserting"    



## End of database stuff


def calculate_perf_metrics(trial_matrix):
    """Calculate simple performance metrics on a session"""
    rec = {}
    
    # Trials and spoiled fraction
    rec['n_trials'] = len(trial_matrix)
    rec['spoil_frac'] = float(np.sum(trial_matrix.outcome == 'spoil')) / \
        len(trial_matrix)

    # Calculate performance
    rec['perf_all'] = float(len(my.pick(trial_matrix, outcome='hit'))) / \
        len(my.pick(trial_matrix, outcome=['hit', 'error']))
    
    # Calculate unforced performance, protecting against low trial count
    n_nonbad_nonspoiled_trials = len(
        my.pick(trial_matrix, outcome=['hit', 'error'], isrnd=True))
    if n_nonbad_nonspoiled_trials < 10:
        rec['perf_unforced'] = np.nan
    else:
        rec['perf_unforced'] = float(
            len(my.pick(trial_matrix, outcome='hit', isrnd=True))) / \
            n_nonbad_nonspoiled_trials

    # Anova with and without remove bad
    for remove_bad in [True, False]:
        # Numericate and optionally remove non-random trials
        numericated_trial_matrix = TrialMatrix.numericate_trial_matrix(
            trial_matrix)
        if remove_bad:
            suffix = '_unforced'
            numericated_trial_matrix = numericated_trial_matrix.ix[
                numericated_trial_matrix.isrnd == True]
        else:
            suffix = '_all'
        
        # Run anova
        aov_res = TrialMatrix._run_anova(numericated_trial_matrix)
        
        # Parse FEV
        if aov_res is not None:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = aov_res['ess'][
                ['ess_prevchoice', 'ess_Intercept', 'ess_rewside']]
        else:
            rec['fev_stay' + suffix], rec['fev_side' + suffix], \
                rec['fev_corr' + suffix] = np.nan, np.nan, np.nan    
    
    return rec



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
    """Plot various overlays
    
    sess_meaned_frames - df with columns 'meaned', 'rewside', 'servo_pos'
    meth -
        'all' - average all with the same rewside together
        'L' - take closest L and furthest R
        'R' - take furthest R and closest L
        'close' - take closest of both
        'far' - take furthest of both
    """
    
    import my.plot
    
    # Split into L and R
    if meth == 'all':
        L = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'left'], axis=0)
        R = np.mean(sess_meaned_frames['meaned'][
            sess_meaned_frames.rewside == 'right'], axis=0)
    elif meth == 'L':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R).irow(0)['meaned']
    elif meth == 'R':
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R).irow(0)['meaned']     
    elif meth == 'close':
        closest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].min()
        closest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].min()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=closest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=closest_R).irow(0)['meaned']     
    elif meth == 'far':
        furthest_L = my.pick_rows(sess_meaned_frames, rewside='left')[
            'servo_pos'].max()            
        furthest_R = my.pick_rows(sess_meaned_frames, rewside='right')[
            'servo_pos'].max()
        L = my.pick_rows(sess_meaned_frames, rewside='left', 
            servo_pos=furthest_L).irow(0)['meaned']
        R = my.pick_rows(sess_meaned_frames, rewside='right', 
            servo_pos=furthest_R).irow(0)['meaned']     
    else:
        raise ValueError("meth not understood: " + str(meth))
            
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

def make_overlays_from_all_fits(overwrite_frames=False, savefig=True):
    """Makes overlays for all available sessions"""
    # Load data
    sbvdf = get_synced_behavior_and_video_df()
    msdf = get_manual_sync_df()
    
    # Join all the dataframes we need
    jdf = sbvdf.join(msdf, on='session', how='right')

    # Do each
    for session in jdf.session:
        make_overlays_from_fits(session, overwrite_frames=overwrite_frames,
            savefig=savefig)

def make_overlays_from_fits(session, overwrite_frames=False, savefig=True):
    """Given a session name, generates overlays and stores in db"""
    # Load data
    sbvdf = get_synced_behavior_and_video_df()
    msdf = get_manual_sync_df()
    
    # Join all the dataframes we need and check that session is in there
    jdf = sbvdf.join(msdf, on='session', how='right')
    metadata = jdf[jdf.session == session]
    if len(metadata) != 1:
        raise ValueError("session %s not found for overlays" % session)
    metadata = metadata.irow(0)
    
    # Dump the frames
    frame_dir = os.path.join(PATHS['database_root'], 'frames', session)
    if not os.path.exists(frame_dir):
        os.mkdir(frame_dir)
        dump_frames_at_retraction_time(metadata, frame_dir)
    elif overwrite_frames:
        dump_frames_at_retraction_time(metadata, frame_dir)

    # Reload the frames
    trial_matrix = get_trial_matrix(session)
    trialnum2frame = load_frames_by_trial(frame_dir, trial_matrix)

    # Keep only those trials that we found images for
    trial_matrix = trial_matrix.ix[sorted(trialnum2frame.keys())]

    # Split on side, servo_pos, stim_number
    res = []
    gobj = trial_matrix.groupby(['rewside', 'servo_pos', 'stepper_pos'])
    for (rewside, servo_pos, stim_number), subti in gobj:
        meaned = np.mean([trialnum2frame[trialnum] for trialnum in subti.index],
            axis=0)
        res.append({'rewside': rewside, 'servo_pos': servo_pos, 
            'stim_number': stim_number, 'meaned': meaned})
    resdf = pandas.DataFrame.from_records(res)

    # Make the various overlays
    import matplotlib.pyplot as plt
    f, axa = plt.subplots(2, 3, figsize=(13, 6))
    make_overlay(resdf, axa[0, 0], meth='all')
    make_overlay(resdf, axa[1, 1], meth='L')
    make_overlay(resdf, axa[1, 2], meth='R')
    make_overlay(resdf, axa[0, 1], meth='close')
    make_overlay(resdf, axa[0, 2], meth='far')
    f.suptitle(session)
    
    # Save or show
    if savefig:
        savename = os.path.join(PATHS['database_root'], 'overlays',
            session + '.png')
        f.savefig(savename)
        plt.close(f)
    else:
        plt.show()    


def dump_frames_at_retraction_time(metadata, session_dir):
    """Dump the retraction time frame into a subdirectory.
    
    metadata : row containing behavior info, video info, and fit info    
    """
    # Load trials info
    trials_info = TrialMatrix.make_trial_matrix_from_file(metadata['filename'])
    splines = TrialSpeak.load_splines_from_file(metadata['filename'])

    # Insert servo retract time
    lines = TrialSpeak.read_lines_from_file(metadata['filename'])
    parsed_df_split_by_trial = \
        TrialSpeak.parse_lines_into_df_split_by_trial(lines)    
    trials_info['time_retract'] = TrialSpeak.identify_servo_retract_times(
        parsed_df_split_by_trial)        

    # Fit to video times
    fit = metadata['fit0'], metadata['fit1']
    video_times = trials_info['time_retract'].values - \
        metadata['guess_vvsb_start']
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
    trials_info = TrialMatrix.make_trial_matrix_from_file(metadata['filename'])
    splines = TrialSpeak.load_splines_from_file(metadata['filename'])
    lines = TrialSpeak.read_lines_from_file(metadata['filename'])
    parsed_df_split_by_trial = \
        TrialSpeak.parse_lines_into_df_split_by_trial(lines)

    # Insert servo retract time
    trials_info['time_retract'] = TrialSpeak.identify_servo_retract_times(
        parsed_df_split_by_trial)

    # Apply the delta-time guess to the retraction times
    test_guess_vvsb = metadata['guess_vvsb_start'] #/ np.timedelta64(1, 's')
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
        return {'test_times': test_times}
    if len(user_results) != N:
        print "warning: len(user_results) should be %d not %d" % (
            N, len(user_results))
        return {'test_times': test_times}
    
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
    
    return {'test_times': test_times, 'resids': resids, 
        'combined_fit': combined_fit}

def search_for_behavior_files(behavior_dir='~/mnt/behave/runmice',
    clean=True):
    """Load behavior files into data frame.
    
    behavior_dir : where to look
    clean : see parse_behavior_filenames
    
    See also search_for_behavior_and_video_files
    """
    # expand path
    behavior_dir = os.path.expanduser(behavior_dir)
    
    # Acquire all behavior files in the subdirectories
    all_behavior_files = []
    for subdir in rigs:
        all_behavior_files += glob.glob(os.path.join(
            behavior_dir, subdir, 'logfiles', 'ardulines.*'))

    # Parse out metadata for each
    behavior_files_df = parse_behavior_filenames(all_behavior_files, 
        clean=clean)    
    
    # Sort and reindex
    behavior_files_df = behavior_files_df.sort('dt_start')
    behavior_files_df.index = range(len(behavior_files_df))
    
    return behavior_files_df

def search_for_behavior_and_video_files(
    behavior_dir='~/mnt/behave/runmice',
    video_dir='~/mnt/bruno-nix/compressed_eye',
    cached_video_files_df=None,
    ):
    """Get a list of behavior and video files, with metadata.
    
    Looks for all behavior directories in behavior_dir/rignumber.
    Looks for all video files in video_dir (using cache).
    Gets metadata about video files using parse_video_filenames.
    Finds which video file maximally overlaps with which behavior file.
    
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

    # Search for behavior files
    behavior_files_df = search_for_behavior_files(behavior_dir)

    # Acquire all video files
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    if len(video_files) == 0:
        print "warning: no video files found"
    video_files_df = parse_video_filenames(video_files, verbose=True,
        cached_video_files_df=cached_video_files_df)

    # Find the best overlap
    new_behavior_files_df = find_best_overlap_video(
        behavior_files_df, video_files_df)
    
    # Join video info
    joined = new_behavior_files_df.join(video_files_df, 
        on='best_video_index', rsuffix='_video')    
    
    return joined, video_files_df

def find_best_overlap_video(behavior_files_df, video_files_df):
    """Find the video file with the best overlap for each behavior file.
    
    Returns : behavior_files_df, but now with a best_video_index and
        a best_video_overlap columns. Suitable for the following:
        behavior_files_df.join(video_files_df, on='best_video_index', 
            rsuffix='_video')
    """
    # Operate on a copy
    behavior_files_df = behavior_files_df.copy()
    
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
        if len(overlap) == 0:
            # ie, no video files found
            continue
        vidx_max_overlap = overlap.argmax()
        
        # Convert from numpy timedelta64 to a normal number
        max_overlap_sec = overlap.ix[vidx_max_overlap] / np.timedelta64(1, 's')
        
        # Store if it's more than zero
        if max_overlap_sec > 0:
            behavior_files_df.loc[bidx, 'best_video_index'] = vidx_max_overlap
            behavior_files_df.loc[bidx, 'best_video_overlap'] = max_overlap_sec

    return behavior_files_df

def parse_behavior_filenames(all_behavior_files, clean=True):
    """Given list of ardulines files, extract metadata and return as df.
    
    Each filename is matched to a pattern which is used to extract the
    rigname, date, and mouse name. Non-matching filenames are discarded.
    
    clean : if True, also clean up the mousenames by upcasing and applying
        aliases. Finally, drop the ones not in the official list of mice.
    """
    # Extract info from filename
    # directory, rigname, datestring, mouse
    pattern = '(\S+)/(\S+)/logfiles/ardulines\.(\d+)\.(\S+)'
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
            rec_l.append({'rig': rig, 'mouse': mouse,
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

    # Add a session name based on the date and cleaned mouse name
    behavior_files_df['session'] = behavior_files_df['filename'].apply(
        lambda s: os.path.split(s)[1].split('.')[1]) + \
        '.' + behavior_files_df['mouse']

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
        if cached_video_files_df is not None and \
            video_filename in cached_video_files_df.filename.values:
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
    
    
    # Sort and reindex
    resdf = resdf.sort('dt_start')
    resdf.index = range(len(resdf))    
    
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