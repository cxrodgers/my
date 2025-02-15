"""This is a catchall module for the stuff that I'm working on that
doesn't fit anywhere else."""
from __future__ import absolute_import

from . import bootstrap
from . import misc
from . import stats
from . import video
from . import dataload

# shortcuts
from .misc import rint, pick, pick_rows, printnow, globjoin