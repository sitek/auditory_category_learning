#!/usr/bin/env python
''' Convert psychopy behavioral logs to BIDS format  '''

import os
import sys
import argparse
import pandas as pd
import numpy as np
from glob import glob

parser = argparse.ArgumentParser(
                description='Generate bids-compatible event file from psychopy log',
                epilog='Example: python convert_behav_to_bids.py --sub=FLT02'
        )

parser.add_argument("--sub", help="participant id", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
#subject_id = 'FLT01'

project_dir = os.path.abspath('/bgfs/bchandrasekaran/krs228/data/FLT/')
behav_dir   = os.path.join(project_dir, 'sourcedata/behav_files/CHA-IBR/')

# psychopy output file names
beh_task_list  = ['ToneLearning', 'STgrid']

# bids task names
bids_task_list = ['tonecat', 'stgrid']

for tx, task_id in enumerate(beh_task_list):
    file_list = sorted(glob(behav_dir + '/*%s*/sub-%s*.csv'%(task_id, subject_id)))
    
    # define initial silence before task begins
    first_acq = 2
    
    # define the gap from start of silent gap to stimulus onset
    #stim_delay = 0.0 # for this subject, this session – confirm for others!
    if 'ToneLearning' in task_id:
        stim_delay = 0.5
    elif 'STgrid' in task_id:
        stim_delay = 0.2

    # define the time before the first stimulus starts
    first_stim_delay = first_acq + stim_delay

    for rx, filename in enumerate(file_list):
        print('converting ', filename)
        fpath = os.path.join(behav_dir, filename)
        df = pd.read_csv(fpath)
        
        # define output path
        out_fpath = os.path.join(project_dir, 'data_bids', 'sub-%s'%subject_id, 'func',
                                 'sub-%s_task-%s_run-%02d_events.tsv'%(subject_id, bids_task_list[tx], rx+1))
        
        # set up dataframe
        bids_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type',
                                        'response_time', 'stim_file', 'feedback'])
        
        # define onset time
        if 'ToneLearning' in task_id:
            bids_df.onset = df['sound_1.started']-(df['sound_1.started'][1]-first_stim_delay)
        elif 'STgrid' in task_id:
            bids_df.onset = df['sound_stimulus.started']-(df['sound_stimulus.started'][1]-first_stim_delay)
        
        # define duration
        #bids_df.duration[df.corrAns>0] = df['sound_1.stopped'] - df['sound_1.started'].astype(np.float16)
        if 'ToneLearning' in task_id:
            bids_df.duration[df.corrAns>0] = 0.3
        elif 'STgrid' in task_id:
            bids_df.duration[df['sound_stimulus.started']>0] = 1.0
        
        # define stimulus type (sound vs. not sound)
        if 'ToneLearning' in task_id:
            bids_df.trial_type[df.soundfile=='stimuli/di1-aN_48000Hz_pol2_S15filt.wav'] = 'di1-aN'
            bids_df.trial_type[df.soundfile=='stimuli/di1-bN_48000Hz_pol2_S15filt.wav'] = 'di1-bN'
            bids_df.trial_type[df.soundfile=='stimuli/di1-hN_48000Hz_pol2_S15filt.wav'] = 'di1-hN'        
            bids_df.trial_type[df.soundfile=='stimuli/di1-iN_48000Hz_pol2_S15filt.wav'] = 'di1-iN'
            bids_df.trial_type[df.soundfile=='stimuli/di2-aN_48000Hz_pol2_S15filt.wav'] = 'di2-aN'
            bids_df.trial_type[df.soundfile=='stimuli/di2-bN_48000Hz_pol2_S15filt.wav'] = 'di2-bN'
            bids_df.trial_type[df.soundfile=='stimuli/di2-hN_48000Hz_pol2_S15filt.wav'] = 'di2-hN'
            bids_df.trial_type[df.soundfile=='stimuli/di2-iN_48000Hz_pol2_S15filt.wav'] = 'di2-iN'
            bids_df.trial_type[df.soundfile=='stimuli/di3-aN_48000Hz_pol2_S15filt.wav'] = 'di3-aN'
            bids_df.trial_type[df.soundfile=='stimuli/di3-bN_48000Hz_pol2_S15filt.wav'] = 'di3-bN'
            bids_df.trial_type[df.soundfile=='stimuli/di3-hN_48000Hz_pol2_S15filt.wav'] = 'di3-hN'
            bids_df.trial_type[df.soundfile=='stimuli/di3-iN_48000Hz_pol2_S15filt.wav'] = 'di3-iN'
            bids_df.trial_type[df.soundfile=='stimuli/di4-aN_48000Hz_pol2_S15filt.wav'] = 'di4-aN'
            bids_df.trial_type[df.soundfile=='stimuli/di4-bN_48000Hz_pol2_S15filt.wav'] = 'di4-bN'
            bids_df.trial_type[df.soundfile=='stimuli/di4-hN_48000Hz_pol2_S15filt.wav'] = 'di4-hN'
            bids_df.trial_type[df.soundfile=='stimuli/di4-iN_48000Hz_pol2_S15filt.wav'] = 'di4-iN'
        elif 'STgrid' in task_id:
            bids_df.trial_type[df['sound_stimulus.started'] > 0]   = 'sound'
            bids_df.trial_type[df.soundFile == 'stimuli/null.wav'] = 'silent'
        
        # define response time (minus stim delay)
        bids_df.response_time = df['key_resp.rt'] - stim_delay
        
        # define stimulus soundfile
        if 'ToneLearning' in task_id:
            bids_df.stim_file = df.soundfile
        elif 'STgrid' in task_id:
            bids_df.stim_file = df.soundFile
        
        # define feedback presented
        if 'ToneLearning' in task_id: 
           bids_df['feedback'] = np.where(df['key_resp.corr']==1, 'right', 
                                          (np.where(df.corrAns==0, 'none', 'wrong')))
        
        # drop the first row if it's not a stimulus
        try:
            # is an error if index has been removed
            pd.isna(bids_df.stim_file[0]) 
            bids_df.drop(axis=0, index=0, inplace=True)
        except:
            pass
        
        # remove null trials (only model stimuli)
        #bids_df = bids_df[bids_df.stim_file != 'stimuli/null.wav']
        
        print(bids_df)
        
        # save to output path
        bids_df.to_csv(out_fpath, sep='\t')
        print('saved output to ', out_fpath)
