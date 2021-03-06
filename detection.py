"""Stuff for Detection task"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import numpy as np
import pandas
import my

def parse_trial_matrix(bfile):
    """
    Here's what happens in Detection.ino.

    1. Repeatedly loop() until lever is pressed long enough to start a trial.
        a.  Lever presses are indicated with "lever press start:", 
            and then a time on the next line, which is kind of useless 
            because it's an int instead of a long.
        b.  If it hasn't been held down for a certain threshold, end loop.
    2.  Start a trial.
        a.  Set trigTime to now, as beginning of trial.
        b.  Turn the lamp off for 25ms, then delay 100ms.
        c.  Print trigTime ("Trial started at: ")
        d.  Determine whether this will be an opto trial.  
            If yes, print "laser ON", then trigger opto and delay 100ms
            If no, print "dark", then delay 100ms
        c.  Choose the trial type
    3.  If a stimulus trial:
        a.  print "stimulus trial"
        b.  Move stimulus into position, with some delays
        c.  Loop until stimDuration ms have elapsed since triggerTime
            i.  If the lever is lifted for long enough, call reward()
            ii. Print "REWARD!!!" and open valve. Various delays here
            iii. break, by inflating elapTime
        d.  Lower optoPin
        e.  Move stim back to rest position
    4.  Otherwise, on catch trial:
        a.  print "catch trial"
        b.  The rest is the same as in #3, except that lifts trigger punish()
            which prints "unrewarded punishment" and delays.
    5.  Delay by ITI
    
    Sync signals:
    Looks like sync is normally high, and goes low for 765 samples (25ms)
    Laser turns on 3780 samples (125ms) after onset of sync pulse
    """
    # Read all the lines
    with file(bfile) as fi:
        lines = fi.readlines()

    # Ignore everything before "BEGIN DATA"
    start_line = np.where(
        [line.strip() == 'BEGIN DATA' for line in lines])[0][0] + 1

    # Parse out commandas and times
    commands = lines[start_line::2]
    times = lines[start_line + 1::2]
    commands = commands[:len(times)] # in case we missed the last line
    times = [int(line.strip()) for line in times]
    commands = [line.strip() for line in commands]
    commands_df = pandas.DataFrame({'command': commands, 'time': times})

    # Drop lever presses
    commands_df = commands_df[commands_df.command != 'lever press start:']
    commands_df.index = list(range(len(commands_df)))

    # Check
    if not np.all(np.sort(commands_df.time.values) == commands_df.time.values):
        print("warning: commands are not sorted")

    # Find when laser started
    laser_start = np.where(commands_df.command.values == 'begin OPTOSTIM')[0][0]

    # Determine the start line of each trial
    trial_start_line_idxs = np.where(
        commands_df['command'] == 'Trial started at:')[0]

    # Form a trial matrix trial by trial
    rec_l = []
    for ntrial in range(len(trial_start_line_idxs) - 1):
        rec = {}

        # Get commands from this trial
        this_trial_start_line = trial_start_line_idxs[ntrial]
        next_trial_start_line = trial_start_line_idxs[ntrial + 1]
        trial_df = commands_df.iloc[this_trial_start_line:next_trial_start_line]
        
        # Determine if we're in the opto period
        after_laser_start_trial = (this_trial_start_line >= laser_start)
        in_laser_start_trial = ((this_trial_start_line < laser_start) and 
            (next_trial_start_line >= laser_start))
        if after_laser_start_trial or in_laser_start_trial:
            rec['warmup'] = False
        else:
            rec['warmup'] = True

        # Parse each command
        zobj = list(zip(trial_df['command'].values, trial_df['time'].values))
        for command, time in zobj:
            if command == 'Trial started at:':
                assert 'start' not in rec
                rec['start'] = old_div(time, 1000.0)
            elif command == 'catch trial':
                assert 'typ' not in rec
                rec['typ'] = 'nogo'
            elif command == 'stimulus trial: bottom rewarded':
                assert 'typ' not in rec
                rec['typ'] = 'go'
            elif command == 'begin OPTOSTIM':
                continue
            elif command == 'unrewarded punishment':
                # This is only for FA
                assert 'outcome' not in rec
                rec['outcome'] = 'FA'
                rec['response_time'] = old_div(time, 1000.0)
            elif command == 'REWARD!!!':
                # This is only for HIT
                assert 'outcome' not in rec
                rec['outcome'] = 'hit'
                rec['response_time'] = old_div(time, 1000.0)
            elif command == 'dark':
                assert 'opto' not in rec
                rec['opto'] = False
            elif command == 'laser ON':
                assert 'opto' not in rec
                rec['opto'] = True
            else:
                old_div(1,0)
        
        # Specify opto explicitly if not yet done (warmup?)
        if 'opto' not in rec:
            rec['opto'] = False
        
        # Specify outcome explicitly on miss and CR
        if 'outcome' not in rec:
            if rec['typ'] == 'go':
                # Was not a HIT, must have been a MISS
                rec['outcome'] = 'miss'
            if rec['typ'] == 'nogo':
                # Was not a FA, must have been a CR
                rec['outcome'] = 'CR'

        # Specify response_time as np.nan unless already set
        if 'response_time' not in rec:
            # Must be a miss or a CR
            assert rec['outcome'] in ['miss', 'CR']
            rec['response_time'] = np.nan
        
        rec_l.append(rec)

    # DataFrame
    trial_matrix = pandas.DataFrame.from_records(rec_l)
    
    # Assign correct
    trial_matrix['correct'] = False
    trial_matrix.loc[trial_matrix.outcome.isin(['hit', 'CR']), 'correct'] = True

    # Error check: only response time can be null
    assert not trial_matrix.drop('response_time', 1).isnull().any().any()
    
    return trial_matrix

def calculate_perf_metrics(trial_matrix, exclude_warmup=True):
    if exclude_warmup:
        trial_matrix = trial_matrix[~trial_matrix.warmup].copy()

    if len(trial_matrix) == 0:
        raise ValueError("no data in trial matrix")
    
    # Perf metrics
    rec_l = []
    for opto in [False, True]:
        for typ in ['go', 'nogo']:
            subdf = my.pick_rows(trial_matrix, opto=opto, typ=typ)
            n_hits = np.sum(subdf.correct)
            n_tots = len(subdf)
            rec_l.append({'opto': opto, 'typ': typ, 'n_hits': n_hits,   
                'n_tots': n_tots})
    perfdf = pandas.DataFrame.from_records(rec_l)
    perfdf['perf'] = old_div(perfdf['n_hits'], perfdf['n_tots'])    
    
    return perfdf
