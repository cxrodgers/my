"""Module for running code on remote server"""
from __future__ import print_function
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
        print("command: %r" % cmd_list)
        sys.stdout.flush()
    
    # Call proc
    proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    returncode = proc.returncode
    
    # Decode
    if stdout is not None:
        stdout = stdout.decode('utf-8')
    if stderr is not None:
        stderr = stderr.decode('utf-8')

    if remove_legal_bullshit:
        stderr = stderr.replace(LEGAL_BULLSHIT, '')

    if announce_stdout:
        print("stdout: %s" % stdout)
        sys.stdout.flush()

    if announce_stderr:
        print("stderr: %s" % stderr)
        sys.stdout.flush()    
    
    if error_on_nonzero_ret and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd_list)

    return cmd_list, stdout, stderr, returncode

def get_now_as_string():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def submit_job(submission_script, verbose=True):
    """Submit a job on habanero
    
    Submits job. If this fails, wait 60s and retry. Continue until success.
    
    submission_script : full path on habanero to script to submit
    verbose : if True, print various status messages
    
    Returns : start_job_result, job_string
        start_job_result : Results of successful job submission
        job_string : The job id as a string
    """
    shell = spur.SshShell(hostname='habanero.rcs.columbia.edu', 
        username='ccr2137')
    with shell:
        # Repeatedly try to submit job (in case slurm error)
        while True:
            # Try to submit the job
            slurm_error = False
            try:
                start_job_result = shell.run(["sbatch", submission_script])
            except spur.RunProcessError:
                # This happens when slurm fails, some socket error
                slurm_error = True
            
            # Depends on if it succeeded
            if slurm_error:
                # Wait so we're not annoying
                if verbose:
                    print("%s  Failed to submit job, waiting to retry" % 
                        get_now_as_string())
                time.sleep(60)
            else:
                # It succeeded, break the loop
                break

        # Try to extract the job id
        try:
            job_string = my.misc.regex_capture('Submitted batch job (\d+)', 
                [start_job_result.output.decode('utf-8')])[0]
        except IndexError:
            raise ValueError("cannot capture job id from %s" % job_string)

    return start_job_result, job_string

def wait_until_job_completes(job_string, verbose=True):
    """Poll habanero until job completes
    
    Returns when job status is "COMPLETED". Raises exception if status
    indicates error or is unknown.
    
    job_string : job id as a string
    
    Returns : probe_job_result
        The result of the successful last probe
    """
    shell = spur.SshShell(hostname='habanero.rcs.columbia.edu', 
        username='ccr2137')
    with shell:
        while True:
            # Extract the job status from an sacct command
            slurm_error = False
            try:
                # squeue is no good because it fails when the job is done
                #~ probe_job_result = shell.run(["squeue", "-j", job_string])
                
                # this one will have two jobs listed after completion, one ending
                # in ".batch", but I think the status will be the same for both
                probe_job_result = shell.run(["sacct", "-j", job_string,
                    '--format=State'])
            except spur.RunProcessError:
                # This happens when slurm fails, some socket error
                slurm_error = True
            
            # Try to parse the status
            if slurm_error:
                status = 'slurm error'
            else:
                pjr_lines = probe_job_result.output.decode('utf-8').split('\n')
                try:
                    status = pjr_lines[2].strip()
                except IndexError:
                    status = 'parse error'
            
            # Announce
            if verbose:
                print("%s  Status: %s" % (get_now_as_string(), status))
            
            # Either keep going, break, or fail
            if status in ['slurm error', 'RUNNING', '', 'PENDING']:
                # keep going
                time.sleep(60)
                continue
            elif status in ['parse error', 'FAILED', 'TIMEOUT', 'DEADLINE', 
                'NODE_FAIL', 'BOOT_FAIL']:
                # a bad error, leave the data and fix me
                raise ValueError("bad error")
            elif status in ['COMPLETED']:
                # success, break
                break
            else:
                raise ValueError("unknown status")
    
    # Successfully completed
    return probe_job_result

