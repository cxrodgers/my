"""Methods for loading shape data"""
import pandas
import os
import my

def load_bwid(params, drop_1_and_6b=True):
    """Load big_waveform_info_df
    
    Loads from params['unit_db_dir']
    Adds stratum
    Drops 1 and 6b
    Load recording_locations_table from params['unit_db_dir']
    Add location_is_strict
    Joins recording_location, crow_recording_location, location_is_strict
    on big_waveform_info_df
    
    Returns: DataFrame
        big_waveform_info_df
    """
    
    ## Load waveform info stuff
    big_waveform_info_df = pandas.read_pickle(
        os.path.join(params['unit_db_dir'], 'big_waveform_info_df'))
    big_waveform_info_df['stratum'] = 'deep'
    big_waveform_info_df.loc[
        big_waveform_info_df.layer.isin(['2/3', '4']), 'stratum'
        ] = 'superficial'

    # Drop 1 and 6b
    if drop_1_and_6b:
        big_waveform_info_df = big_waveform_info_df.loc[
            ~big_waveform_info_df['layer'].isin(['1', '6b'])
            ].copy()
    
    # Remove those levels
    big_waveform_info_df.index = (
        big_waveform_info_df.index.remove_unused_levels())


    ## Join recording location
    # Load and rename
    recording_locations_table = pandas.read_csv(
        os.path.join(params['unit_db_dir'], 
        '20191007 electrode locations - Sheet1.csv')).rename(columns={
        'Session': 'session', 'Closest column': 'recording_location', 
        'Closest C-row column': 'crow_recording_location', 
        }).set_index('session').sort_index()

    # fillna the really off-target ones
    recording_locations_table['crow_recording_location'] = (
        recording_locations_table['crow_recording_location'].fillna('off'))

    # Add a "strict" column where the recording was bona fide C-row
    recording_locations_table['location_is_strict'] = (
        recording_locations_table['recording_location'] ==
        recording_locations_table['crow_recording_location'])

    # Join onto bwid
    big_waveform_info_df = big_waveform_info_df.join(recording_locations_table[
        ['recording_location', 'crow_recording_location', 'location_is_strict']
        ], on='session')
    
    # Error check
    assert not big_waveform_info_df.isnull().any().any()
    
    return big_waveform_info_df
    
def load_session_metadata(params):
    """Load metadata about sessions, tasks, and mice.
    
    Returns: tuple
        session_df, task2mouse, mouse2task
    """
    session_df = pandas.read_pickle(
        os.path.join(params['pipeline_dir'], 'session_df'))
    task2mouse = session_df.groupby('task')['mouse'].unique()
    mouse2task = session_df[
        ['task', 'mouse']].drop_duplicates().set_index('mouse')['task']
    
    return session_df, task2mouse, mouse2task

def load_big_tm(params, dataset='no_opto', mouse2task=None):
    """Load big_tm, the big trial matrix, and optionally filters.
    
    params : parameters from json file
    
    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns original big_tm.
    
    mouse2task : Series, or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to big_tm index.
        If None, does nothing.
    
    Returns: DataFrame
        big_tm    
    """
    # Load original big_tm with all trials
    big_tm = pandas.read_pickle(
        os.path.join(params['patterns_dir'], 'big_tm'))

    # Slice out the trials of this dataset (no_opto) from big_tm
    if dataset is not None:
        included_trials = pandas.read_pickle(
            os.path.join(params['logreg_dir'], 'datasets', dataset, 'labels')
            ).index
        
        # Apply mask
        big_tm = big_tm.loc[included_trials]
        big_tm.index = big_tm.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        big_tm = my.misc.insert_mouse_and_task_levels(
            big_tm, mouse2task)
    
    return big_tm


def load_data_from_patterns(params, filename, dataset='no_opto', 
    mouse2task=None):
    """Common loader function from patterns dir
    
    filename : string
        These are the valid options:
            big_tm
            big_C2_tip_whisk_cycles
            big_cycle_features
            big_touching_df
            big_tip_pos
            big_grasp_df

        These are unsupported, because they aren't indexed the same:
            big_ccs_df
            kappa_parameterized
            peri_contact_kappa
    
    params : parameters from json file

    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns original big_tm.
    
    mouse2task : Series, or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to big_tm index.
        If None, does nothing.
    
    Returns: DataFrame
        The requested data.
    """
    # Load from patterns directory
    full_filename = os.path.join(params['patterns_dir'], filename)
    
    # Special case loading
    if filename == 'big_tip_pos':
        res = pandas.read_hdf(full_filename)
    else:
        res = pandas.read_pickle(full_filename)
    
    # Slice out the trials of this dataset (no_opto)
    if dataset is not None:
        # Load trials
        included_trials = pandas.read_pickle(
            os.path.join(params['logreg_dir'], 'datasets', dataset, 'labels')
            ).index
        
        # Apply mask
        res = my.misc.slice_df_by_some_levels(res, included_trials)
        res.index = res.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        res = my.misc.insert_mouse_and_task_levels(res, mouse2task)
    
    return res
    

def load_data_from_logreg(params, filename, dataset='no_opto', mouse2task=None):
    """Load data from logreg directory
    
    filename : string
        These are the valid options:
            unobliviated_unaggregated_features
            unobliviated_unaggregated_features_with_bin
            obliviated_aggregated_features
            obliviated_unaggregated_features_with_bin
        
        These are unsupported:
            BINS
        
    params : parameters from json file

    dataset : string or None
        If string, loads corresponding dataset, and includes only those
        trials in the result.
        If None, returns without filtering.
        
        If filename == 'obliviated_aggregated_features' and dataset is not None,
        then the pre-sliced version is loaded from the dataset directory.
    
    mouse2task : Series or None
        If Series (from load_session_metadat), then adds mouse and task
        levels to index.
        If None, does nothing.
    
    Returns: DataFrame
        The requested data.    
    """
    # Load, depending on filename
    if filename == 'oblivated_aggregated_features' and dataset is not None:
        # Special case: this was already sliced and dumped in the dataset dir
        full_filename = os.path.join(
            params['logreg_dir'], 'datasets', dataset, 'features')
        res = pandas.read_pickle(full_filename)
    
    else:
        # Load
        full_filename = os.path.join(params['logreg_dir'], filename)
        res = pandas.read_pickle(full_filename)

        # Slice out the trials of this dataset (no_opto)
        if dataset is not None:
            # Load trials
            included_trials = pandas.read_pickle(os.path.join(
                params['logreg_dir'], 'datasets', dataset, 'labels')
                ).index
            
            # Apply mask
            res = my.misc.slice_df_by_some_levels(res, included_trials)
            res.index = res.index.remove_unused_levels()

    # Insert mouse and task levels
    if mouse2task is not None:
        res = my.misc.insert_mouse_and_task_levels(res, mouse2task)
    
    return res