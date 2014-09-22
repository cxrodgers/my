"""Module for parsing behavior files and video"""
import os, numpy as np, glob, re, pandas, datetime
import misc
import subprocess # for ffprobe

# Known mice
mice = ['AM03', 'AM05', 'KF13', 'KM14', 'KF16', 'KF17', 'KF18', 'KF19', 'KM24', 'KM25']
aliases = {
    'KF13A': 'KF13',
    'AM03A': 'AM03',
    }
assert np.all([alias_val in mice for alias_val in aliases.values()])


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

    if clean:
        # Clean the behavior files by upcasing and applying aliases
        behavior_files_df.mouse = behavior_files_df.mouse.apply(str.upper)
        behavior_files_df.mouse.replace(aliases, inplace=True)

        # Drop any that are not in the list of accepted mouse names
        behavior_files_df = behavior_files_df.ix[behavior_files_df.mouse.isin(mice)]

    return behavior_files_df

def parse_video_filenames(video_filenames, verbose=False):
    """Given list of video files, extract metadata and return df.

    For each filename, we extract the date (from the filename) and duration
    (using ffprobe).
    """
    # Extract info from filename
    # directory, rigname, datestring, extension
    pattern = '(\S+)/(\S+)\.(\d+)\.(\S+)'
    rec_l = []

    for video_filename in video_filenames:
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