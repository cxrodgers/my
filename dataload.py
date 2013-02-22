"""Methods for loading data from my LBPB experiments

This is all specific to the layout of the data on my computer and
the typical defaults for this expt.
"""

import numpy as np
import pandas, kkpandas, kkpandas.kkrs
import os.path
from lxml import etree
from ns5_process import LBPB, RecordingSession


def ulabel2dfolded(ulabel, folding_kwargs=None, trial_picker_kwargs='random hits'):
    """Convenience function for getting dict of folded from RS/kkpandas
    
    Some reasonable defaults for kwargs ... see code  
    
    Returns: dict, picked trials label to Folded object
    """
    # Default kwargs for the pipeline
    # How to parse out trials
    if trial_picker_kwargs == 'random hits':
        trial_picker_kwargs = {
            'labels': LBPB.mixed_stimnames,
            'label_kwargs': [{'stim_name':s} for s in LBPB.mixed_stimnames],
            'nonrandom' : 0,
            'outcome' : 'hit'
            }
    elif trial_picker_kwargs == 'all':
        trial_picker_kwargs = {
            'labels': LBPB.stimnames, 
            'label_kwargs': [{'stim_name':s} for s in LBPB.stimnames],
            }
    elif trial_picker_kwargs == 'by outcome':
        label_kwargs = pandas.MultiIndex.from_tuples(
            names=['stim_name', 'outcome'],
            tuples=list(itertools.product(
                LBPB.mixed_stimnames, ['hit', 'error', 'wrong_port'])))
        labels = ['-'.join(t) for t in label_kwargs]
        trial_picker_kwargs = {'labels': labels, 'label_kwargs': label_kwargs,
            'nonrandom' : 0}

    # How to fold the window around each trial
    if folding_kwargs is None:
        folding_kwargs = {'dstart': -.25, 'dstop': .3}    


    # Load data
    gets = getstarted()

    # Parse ulabel
    session_name = kkpandas.kkrs.ulabel2session_name(ulabel)
    unum = kkpandas.kkrs.ulabel2unum(ulabel)

    # link back
    rs, kks = session2rs(session_name), session2kk_server(session_name)

    # Run the pipeline
    res = kkpandas.pipeline.pipeline_overblock_oneevent(
        kks, session_name, unum, rs,
        trial_picker_kwargs=trial_picker_kwargs,
        folding_kwargs=folding_kwargs)

    return res



def getstarted():
    """Load all my data into kkpandas and RS objects
    
    Returns: dict with following items:
        xmlfiles : dict, ratname to xml file
        kksfiles : dict, ratname to kk_server file
        kk_servers : dict, ratname to kk_server object
        xml_roots : dict, ratname to XML root object
        data_dirs : dict, ratname to location of data
        manual_units : dict, ratname to manually sorted units (XML objects)
            Includes only units from XML files with score above 3 and
            with session marked analyze=True
        unit_db : pandas DataFrame consisting of information about each ulabel
    """
    res = {}
    
    res['xmlfiles'] = {
        'CR20B' : os.path.expanduser('~/Dropbox/lab/CR20B_summary/CR20B.xml'),
        'CR21A' : os.path.expanduser('~/Dropbox/lab/CR21A_summary/CR21A.xml'),
        'YT6A' : os.path.expanduser('~/Dropbox/lab/YT6A_summary/YT6A.xml'),
        'CR12B' : os.path.expanduser('~/Dropbox/lab/CR12B_summary_v2/CR12B.xml'),
        'CR17B' : os.path.expanduser('~/Dropbox/lab/CR17B_summary_v2/CR17B.xml'),
        'CR24A' : os.path.expanduser('~/Dropbox/lab/CR24A_summary/CR24A.xml'),
        }
    
    res['kksfiles'] = {
        'CR20B' : os.path.expanduser(
            '~/Dropbox/lab/CR20B_summary/CR20B_behaving.kks'),
        'CR21A' : os.path.expanduser(
            '~/Dropbox/lab/CR21A_summary/CR21A_behaving.kks'),
        'YT6A' : os.path.expanduser(
            '~/Dropbox/lab/YT6A_summary/YT6A_behaving.kks'),
        'CR17B' : os.path.expanduser(
            '~/Dropbox/lab/CR17B_summary_v2/CR17B_behaving.kks'),
        'CR12B' : os.path.expanduser(
            '~/Dropbox/lab/CR12B_summary_v2/CR12B_behaving.kks'),            
        'CR24A' : os.path.expanduser(
            '~/Dropbox/lab/CR24A_summary/CR24A_behaving.kks'),            
        }
    
    res['kk_servers'] = dict([
        (ratname, kkpandas.kkio.KK_Server.from_saved(kksfile))
        for ratname, kksfile in res['kksfiles'].items()])
    
    res['data_dirs'] = {
        'CR20B' : '/media/hippocampus/chris/20120705_CR20B_allsessions',
        'CR21A' : '/media/hippocampus/chris/20120622_CR21A_allsessions',
        'YT6A' : '/media/hippocampus/chris/20120221_YT6A_allsessions',
        'CR17B' : '/media/hippocampus/chris/20121220_CR17B_allsessions',
        'CR12B' : '/media/hippocampus/chris/20121115_CR12B_allsessions',
        'CR24A' : '/media/hippocampus/chris/20121217_CR24A_allsessions',
        #'CR12B' : '/media/granule/20121115_CR12B_allsessions',
        #'CR12B' : '/media/hippocampus/chris/20111208_CR12B_allsessions_sorted',
        }
    
    res['xml_roots'] = dict([
        (ratname, etree.parse(xmlfile).getroot())
        for ratname, xmlfile in res['xmlfiles'].items()])

    xpath_str = '//unit[quality/text()>=3 and ../../../@analyze="True"]'
    res['manual_units'] = dict([
        (ratname, root.xpath(xpath_str))
        for ratname, root in res['xml_roots'].items()])
    
    res['unit_db'] = pandas.DataFrame.from_csv(os.path.expanduser(
        '~/Dropbox/lab/unit_db.csv'))

    return res

# Linking functions between RS and kkpandas objects that are specific
# to my data
def session2rs(session_name):
    gets = getstarted()
    kk_servers, data_dirs = gets['kk_servers'], gets['data_dirs']
    
    for ratname, kk_server in kk_servers.items():
        if session_name not in kk_server.session_list:
            continue
        
        # Session found
        data_dir = data_dirs[ratname]
        rs = RecordingSession.RecordingSession(
            os.path.join(data_dir, session_name))
        
        return rs
    
    # No session ever found
    raise ValueError("No session like %s found!" % session_name)

def session2kk_server(session_name):
    gets = getstarted()
    kk_servers = gets['kk_servers']
    
    for ratname, kk_server in kk_servers.items():
        if session_name in kk_server.session_list:
            return kk_server
        
    raise ValueError("No session like %s found!" % session_name)