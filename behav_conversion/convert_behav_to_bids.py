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

    run_i = 1
    for rx, filename in enumerate(file_list):
        #try:
        print('converting ', filename)
        fpath = os.path.join(behav_dir, filename)
        df = pd.read_csv(fpath)

        # create a temp dataframe of only trials where sounds were presented
        trial_df = df[df.corrAns>0]

        if len(trial_df)<30:
            print('too few trials – incomplete run. Skipping')
        else:
            ''' Stimulus dataframe '''
            # set up stimulus dataframe
            stim_df = pd.DataFrame(columns=['onset', 
                                            'duration', 
                                            'trial_type',
                                            'stim_file'])

            # define onset time (relative to the first stimulus presentation)
            stim_df.onset = trial_df['sound_1.started'] - \
                            (trial_df['sound_1.started'].iloc[0]-first_stim_delay)

            # define duration
            stim_df.duration = 0.3

            # define stimulus type (based on sound file – HARDCODED)
            stim_df.trial_type = 'sound_'+trial_df.soundfile.str[8:14]

            # define stimulus soundfile
            stim_df.stim_file = trial_df.soundfile

            ''' Response dataframe '''
            # set up response dataframe
            resp_df = pd.DataFrame(columns=['onset', 
                                            'duration',
                                            'response_time', 
                                            'correct_key',
                                            'trial_type'])

            # define onset time (relative to the first stimulus presentation)
            resp_df.onset = trial_df['sound_1.started'] + \
                            trial_df['key_resp.rt']  - \
                            (trial_df['sound_1.started'].iloc[0]-first_stim_delay)

            # define duration (arbitrary)
            resp_df.duration = 0.5

            resp_df.response_time = trial_df['key_resp.rt']        
            resp_df.correct_key = trial_df['corrAns']
            resp_df.trial_type = 'resp_'+trial_df['key_resp.keys'].astype(str)

            ''' Feedback dataframe '''
            # set up feedback dataframe
            fb_df = pd.DataFrame(columns=['onset',
                                            'duration', 
                                            'trial_type'])        

            # define onset time (relative to the first stimulus presentation)
            fb_df.onset = trial_df['text_2.started'] - \
                          (trial_df['sound_1.started'].iloc[0]-first_stim_delay)

            # feedback is visible from the onset of text_2 to the onset of jitter_cross_post_fb
            fb_df.duration = trial_df['jitter_cross_post_fb.started'] - \
                             trial_df['text_2.started']

            # define feedback presented
            # TO DO: UPDATE TO NOT INCLUDE NULL TRIALS IN FB_WRONG
            #fb_df['trial_type'] = np.where(trial_df['key_resp.corr']==1, 'fb_correct', 
            #                                (np.where(trial_df.corrAns==0, 'none', 'fb_wrong')))
            cond_list = [trial_df['key_resp.corr']==1, # correct response
                         trial_df['key_resp.keys']=='None', # no response
                         trial_df.corrAns==0, # null stimulus
                         trial_df['key_resp.corr']==0 # wrong response (must be after no response cond)
                        ]
            choice_list = ['fb_correct',
                           'fb_noresp',
                           'fb_none',
                           'fb_wrong'
                          ]

            fb_df['trial_type'] = np.select(cond_list, 
                                            choice_list,
                                            'none')

            ''' combine all three dataframes '''
            bids_df = pd.concat([stim_df, resp_df, fb_df], 
                                axis=0, join='outer', ignore_index=True)
            bids_df.sort_values(by=['onset'], ignore_index=True,
                                inplace=True)

            # save to output path
            out_fpath = os.path.join(project_dir,
                                     'data_denoised',
                                     'sub-%s'%subject_id, 'func',
                                     'sub-%s_task-%s_run-%02d_acq-dwidenoise_events.tsv'%(subject_id, bids_task_list[0], run_i))

            bids_df.to_csv(out_fpath, sep='\t')
            print('saved output to ', out_fpath)
            run_i += 1
        #except:
        #    print('could not process this csv file')
        #    pass


''' Spectrotemporal grid stimulus task '''
if 'STgrid' in task_id:

    # define initial BOLD acquisition time before task begins during silent gap
    first_acq = 2
    stim_delay = 0.4

    # define the time before the first stimulus starts
    first_stim_delay = first_acq + stim_delay

    for rx, filename in enumerate(file_list):
        print('converting ', filename)
        fpath = os.path.join(behav_dir, filename)
        df = pd.read_csv(fpath)

        if len(df) < 100:
            print('too few trials. skipping')
        else:
            # define output path
            out_fpath = os.path.join(project_dir,
                                     'data_denoised',
                                     'sub-%s'%subject_id, 'func',
                                     'sub-%s_task-%s_run-%02d_acq-dwidenoise_events.tsv'%(subject_id, 
                                                                           bids_task_list[1], 
                                                                           rx+1))

            # set up dataframe
            bids_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type', 
                                            'temp_mod_rate', 'spect_mod_rate',
                                            'response_time'])

            #bids_df.onset = df['sound_stimulus.started']-(df['sound_stimulus.started'][1]-first_stim_delay)
            #bids_df.duration[df['sound_stimulus.started']>0] = 1.0

            onset_list = []
            stim_list = []
            temp_list = []
            spect_list = []
            for sx, stim in enumerate(df.soundFile):
                if isinstance(stim, str) and 'S15' in stim:
                    if stim != df.soundFile[sx-1]:
                        stim_num = int(stim.split('_')[3])
                        temp_mod = mod_df.loc[stim_num]['temp_mod_rate']
                        spect_mod = mod_df.loc[stim_num]['spect_mod_rate']

                        onset = (df['sound_stimulus.started'][sx]-
                                 (df['sound_stimulus.started'][1]-first_stim_delay))
                        onset_list.append(onset)

                        stim_list.append('stim%02d'%stim_num)
                        temp_list.append(temp_mod)
                        spect_list.append(spect_mod)

            bids_df.onset = onset_list        
            bids_df.trial_type = stim_list
            bids_df.temp_mod_rate = temp_list
            bids_df.spect_mod_rate = spect_list
            bids_df.duration = 20 
            print(bids_df)

            # save to output path
            bids_df.to_csv(out_fpath, sep='\t')
            print('saved output to ', out_fpath)