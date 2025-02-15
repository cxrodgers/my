"""Functions for detecting onsets and syncrhronizing events"""
# Moved from MCwatch.behavior.syncing
# Moved to paclab.syncing

import numpy as np
import pandas

from paclab.syncing import (
    extract_onsets_and_durations, drop_refrac, extract_duration_of_onsets,
    extract_duration_of_onsets2, longest_unique_fit,
    )

print('TODO: replace all calls to my.syncing with paclab.syncing')
