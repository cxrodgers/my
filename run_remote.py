"""Module for running code on remote server"""
import os
import subprocess
import spur
import my
import time
import datetime
import pandas
import numpy as np
import sys

LEGAL_BULLSHIT = ('The information in University Systems at Columbia '
    'University is private and\nconfidential and may be used only on a '
    'need-to-know basis. All access is logged.\nUnauthorized or improper use '
    'of a University System or the data in it may result\nin dismissal '
    'and/or civil or criminal penalties.\n\n')

def run_rsync(src, dst, flags=None, announce_cmd=True, announce_stdout=True,
    announce_stderr=True, error_on_nonzero_ret=True,
    remove_legal_bullshit=True):
    """Call rsync in subprocess and return result"""
    # default flags
    if flags is None:
        flags = ['-rtv']
    
    # Concatenate tokens into list
    cmd_list = ['rsync'] + flags + [src, dst]
    
    # Announce command
    if announce_cmd:
        print "command: %r" % cmd_list
        sys.stdout.flush()
    
    # Call proc
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    returncode = proc.returncode

    if remove_legal_bullshit:
        stderr = stderr.replace(LEGAL_BULLSHIT, '')

    if announce_stdout:
        print "stdout: %s" % stdout
        sys.stdout.flush()

    if announce_stderr:
        print "stderr: %s" % stderr
        sys.stdout.flush()    
    
    if error_on_nonzero_ret and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd_list)

    return cmd_list, stdout, stderr, returncode

def get_now_as_string():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')