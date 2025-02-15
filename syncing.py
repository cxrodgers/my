"""Functions for detecting onsets and syncrhronizing events

Moved from MCwatch.behavior.syncing
Moved to paclab.syncing

This file is now deprecated - replaced with paclab.syncing
"""

import numpy as np
import pandas

def extract_onsets_and_durations(*args, **kwargs):
    import paclab.syncing

    print(
        'warning: replace all calls to my.syncing.extract_onsets_and_durations '
        'with paclab.syncing.extract_onsets_and_durations instead'
        )
    return paclab.syncing.extract_onsets_and_durations(*args, **kwargs)

def drop_refrac(*args, **kwargs):
    import paclab.syncing

    print(
        'warning: replace all calls to my.syncing.drop_refrac '
        'with paclab.syncing.drop_refrac instead'
        )
    return paclab.syncing.drop_refrac(*args, **kwargs)

def extract_duration_of_onsets(*args, **kwargs):
    import paclab.syncing

    print(
        'warning: replace all calls to my.syncing.extract_duration_of_onsets '
        'with paclab.syncing.extract_duration_of_onsets instead'
        )
    return paclab.syncing.extract_duration_of_onsets(*args, **kwargs)

def extract_duration_of_onsets2(*args, **kwargs):
    import paclab.syncing

    print(
        'warning: replace all calls to my.syncing.extract_duration_of_onsets2 '
        'with paclab.syncing.extract_duration_of_onsets2 instead'
        )
    return paclab.syncing.extract_duration_of_onsets2(*args, **kwargs)

def longest_unique_fit(*args, **kwargs):
    import paclab.syncing

    print(
        'warning: replace all calls to my.syncing.longest_unique_fit '
        'with paclab.syncing.longest_unique_fit instead'
        )
    return paclab.syncing.longest_unique_fit(*args, **kwargs)
