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
                epilog='Example: python convert_behav_to_bids.py --sub FLT02 --task ToneLearning'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--task", help="task id (options: 'ToneLearning', 'STgrid'", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_id = args.task

project_dir = os.path.abspath('/bgfs/bchandrasekaran/krs228/data/FLT/')
behav_dir   = os.path.join(project_dir, 'sourcedata', 'behav_files', 'CHA-IBR/')
#project_dir = os.path.join('/Users/krs228', 'data', 'FLT')
#behav_dir = os.path.join('/Users/krs228/','OneDrive - University of Pittsburgh/','CHA-IBR/')

# bids task names
bids_task_list = ['tonecat', 'stgrid']

file_list = sorted(glob(behav_dir + '/*%s*/sub-%s*.csv'%(task_id, subject_id)))

# define initial BOLD acquisition time before task begins during silent gap
first_acq = 2

''' ToneLearning task '''
if 'ToneLearning' in task_id:
    # in this task, stimuli start 0.5 s after the silent gap starts
    stim_delay = 0.5

    # define the time before the first stimulus starts
    first_stim_delay = first_acq + stim_delay

    for rx, filename in enumerate(file_list):
        print('converting ', filename)
        fpath = os.path.join(behav_dir, filename)
        df = pd.read_csv(fpath)
        
        # define output path
        out_fpath = os.path.join(project_dir,
                                 'data_bids', 
                                 'sub-%s'%subject_id, 'func',
                                 'sub-%s_task-%s_run-%02d_events.tsv'%(subject_id, bids_task_list[0], rx+1))
        
        # create a temp dataframe of only trials where sounds were presented
        trial_df = df[df.corrAns>0]

        ''' Stimulus dataframe '''
        # set up stimulus dataframe
        stim_df = pd.DataFrame(columns=['onset', 
                                        'duration', 
                                        'trial_type',
                                        'stim_file'])
        
        # define onset time (relative to the first stimulus presentation)
        stim_df.onset = trial_df['sound_1.started'] - (trial_df['sound_1.started'].iloc[0]-first_stim_delay)
        
        # define duration
        #stim_df.duration = trial_df['sound_1.stopped'].astype(np.float16) - trial_df['sound_1.started'].astype(np.float16)
        stim_df.duration = 0.3
        
        # define stimulus type (based on sound file – HARDCODED)
        stim_df.trial_type = trial_df.soundfile.str[8:14]
        '''
        stim_df.trial_type[trial_df.soundfile=='stimuli/di1-aN_48000Hz_pol2_S15filt.wav'] = 'di1-aN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di1-bN_48000Hz_pol2_S15filt.wav'] = 'di1-bN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di1-hN_48000Hz_pol2_S15filt.wav'] = 'di1-hN'        
        stim_df.trial_type[trial_df.soundfile=='stimuli/di1-iN_48000Hz_pol2_S15filt.wav'] = 'di1-iN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di2-aN_48000Hz_pol2_S15filt.wav'] = 'di2-aN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di2-bN_48000Hz_pol2_S15filt.wav'] = 'di2-bN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di2-hN_48000Hz_pol2_S15filt.wav'] = 'di2-hN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di2-iN_48000Hz_pol2_S15filt.wav'] = 'di2-iN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di3-aN_48000Hz_pol2_S15filt.wav'] = 'di3-aN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di3-bN_48000Hz_pol2_S15filt.wav'] = 'di3-bN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di3-hN_48000Hz_pol2_S15filt.wav'] = 'di3-hN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di3-iN_48000Hz_pol2_S15filt.wav'] = 'di3-iN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di4-aN_48000Hz_pol2_S15filt.wav'] = 'di4-aN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di4-bN_48000Hz_pol2_S15filt.wav'] = 'di4-bN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di4-hN_48000Hz_pol2_S15filt.wav'] = 'di4-hN'
        stim_df.trial_type[trial_df.soundfile=='stimuli/di4-iN_48000Hz_pol2_S15filt.wav'] = 'di4-iN'
        '''

        # define stimulus soundfile
        stim_df.stim_file = trial_df.soundfile.str[8:14]

        ''' Response dataframe '''
        # set up response dataframe
        resp_df = pd.DataFrame(columns=['onset', 
                                        'duration',
                                        'response_time', 
                                        'correct_key',
                                        'response_key'])
        
        # define onset time (relative to the first stimulus presentation)
        resp_df.onset = trial_df['sound_1.started'] + trial_df['key_resp.rt']  - (trial_df['sound_1.started'].iloc[0]-first_stim_delay)
        
        # define duration (arbitrary)
        resp_df.duration = 0.1

        resp_df.response_time = trial_df['key_resp.rt']        
        resp_df.correct_key = trial_df['corrAns']
        resp_df.response_key = trial_df['key_resp.keys']

        ''' Feedback dataframe '''
        # set up feedback dataframe
        fb_df = pd.DataFrame(columns=['onset',
                                        'duration', 
                                        'feedback'])        
        
        # define onset time (relative to the first stimulus presentation)
        fb_df.onset = trial_df['text_2.started'] - (trial_df['sound_1.started'].iloc[0]-first_stim_delay)

        # feedback is visible from the onset of text_2 to the onset of jitter_cross_post_fb
        fb_df.duration = trial_df['jitter_cross_post_fb.started'] - trial_df['text_2.started']

        # define feedback presented
        fb_df['feedback'] = np.where(trial_df['key_resp.corr']==1, 'correct', 
                                        (np.where(trial_df.corrAns==0, 'none', 'wrong')))

        ''' combine all three dataframes '''
        bids_df = pd.concat([stim_df, resp_df, fb_df], 
                            axis=0, join='outer', ignore_index=True)
        bids_df.sort_values(by=['onset'], ignore_index=True,
                            inplace=True)

        #print(bids_df)
        
        # save to output path
        bids_df.to_csv(out_fpath, sep='\t')
        print('saved output to ', out_fpath)


### WORK IN PROGRESS
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
                                    'sub-%s_task-%s_run-%02d_events.tsv'%(subject_id, bids_task_list[1], rx+1))
        
        # set up dataframe
        bids_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type',
                                        'response_time', 'stim_file', 'feedback'])
        
        bids_df.onset = df['sound_stimulus.started']-(df['sound_stimulus.started'][1]-first_stim_delay)
        bids_df.duration[df['sound_stimulus.started']>0] = 1.0

        bids_df.trial_type[df['sound_stimulus.started'] > 0]   = 'sound'
        bids_df.trial_type[df.soundFile == 'stimuli/null.wav'] = 'silent'
    
        # define response time (minus stim delay)
        bids_df.response_time = df['key_resp.rt'] - stim_delay
        
        bids_df.stim_file = df.soundFile
    
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
