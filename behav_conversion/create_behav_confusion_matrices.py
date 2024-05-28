import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from seaborn import heatmap

project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')

bidsroot = os.path.join(project_dir, 'data_denoised')
deriv_dir = os.path.join(bidsroot, 'derivatives')
fmriprep_dir = os.path.join(deriv_dir, 'denoised_fmriprep-22.1.1')

beh_out_dir = os.path.join(deriv_dir, 'behavior')

task_list = ['tonecat']
task_label = task_list[0]

participants_fpath = os.path.join(bidsroot, 'participants.tsv')
participants_df = pd.read_csv(participants_fpath, sep='\t')

# subjects to ignore (not fully processed, etc.)
ignore_subs = ['sub-FLT27',
               'sub-FLT07', # bad QA 11/14/23
               'sub-FLT02', # missing resp_6 in run00 (pressing wrong keys)
               #'sub-FLT10', # MISSING 11/16/23
               #'sub-FLT01', 'sub-FLT16',  
               #'sub-FLT19', 'sub-FLT20',
               #'sub-FLT28', 'sub-FLT30',
              ]
participants_df.drop(participants_df[participants_df.participant_id.isin(ignore_subs)].index, inplace=True)

# re-sort by participant ID
participants_df.sort_values(by=['participant_id'], ignore_index=True, inplace=True)

# create group-specific lists of subject IDs
sub_list_mand = list(participants_df.participant_id[participants_df.group=='Mandarin'])
sub_list_nman = list(participants_df.participant_id[participants_df.group=='non-Mandarin'])
sub_dict = {'Mandarin': sub_list_mand, 'non-Mandarin': sub_list_nman}
participant_list = sub_list_mand + sub_list_nman

for sub_id in participant_list:
    print(sub_id)
    
    # make output directory
    sub_beh_out_dir = os.path.join(beh_out_dir, sub_id)
    os.makedirs(sub_beh_out_dir, exist_ok=True)
        
    sub_bids_dir = os.path.join(bidsroot, sub_id, 'func')
    bids_tsv_list = sorted(glob(sub_bids_dir+f'/{sub_id}*{task_label}*.tsv'))

    for run_tsv in bids_tsv_list:
        tsv_pd = pd.read_csv(run_tsv, sep='\t')
        print(tsv_pd.head())

        tsv_pd = tsv_pd[~tsv_pd['trial_type'].str.contains('resp_None')]

        run_id = os.path.basename(run_tsv).split('_')[2]
        print(run_id)

        ''' Tone confusion matrix '''
        simple_df = tsv_pd.loc[:,['trial_type', 'correct_key']].dropna()
        simple_df = simple_df[~tsv_pd['trial_type'].str.contains('resp_8')]

        simple_df['correct_key'] = simple_df['correct_key'].astype(int).astype(str)

        simple_df['trial_type'] = [x[-1] for x in simple_df['trial_type']]

        renum_simple_df = simple_df.astype(str).replace(['7', '6', '1', '2'], 
                                                        ['1', '2', '3', '4'])

        cm = confusion_matrix(renum_simple_df['correct_key'], renum_simple_df['trial_type'], normalize='true')
        
        # save category x category confusion matrix
        out_fpath = os.path.join(sub_beh_out_dir, f'{sub_id}_{run_id}_tonecat_confusion_matrix.tsv')
        np.savetxt(out_fpath, cm, delimiter = '\t')

        print('sub-{} {} tone accuracy = {:.03f}'.format(sub_id, run_id, cm.diagonal().mean()))

        ''' Stimulus-based behavioral RDMs '''
        stim_df = tsv_pd.loc[:,['trial_type', 'correct_key']]

        # create a `stimulus` row
        stim_df['stimulus'] = ''

        # copy stimulus info to the response rows
        for ix in range(1, len(stim_df)):
            if 'resp' in stim_df['trial_type'][ix]:
                stim_df['stimulus'].iloc[ix] = stim_df['trial_type'].iloc[ix-1].split('_')[1]

        stim_df.dropna(inplace=True)

        stim_df['trial_type'] = [x[-1] for x in stim_df['trial_type']]

        stim_df['trial_type'] = stim_df['trial_type'].astype(str).replace(['7', '6', '1', '2'], 
                                                                          ['1', '2', '3', '4'])

        stim_df['correct_key'] = stim_df['correct_key'].replace([7.0, 6.0, 1.0, 2.0], 
                                                                ['1', '2', '3', '4'])

        stim_list = np.unique(stim_df[stim_df['stimulus'].str.contains('di')]['stimulus'])

        # extract responses for each stimulus
        all_stim_responses = []
        for sx, stim in enumerate(stim_list):
            stim_responses = []
            for block in range(3):
                try:
                    stim_responses.append(stim_df.loc[48*(block):48*(block+1)][stim_df['stimulus']==stim]['trial_type'].array[0])
                except:
                    stim_responses.append(np.nan)
            all_stim_responses.append(stim_responses)

        # compare responses across stimuli
        n_stim = len(all_stim_responses)
        stim_conf_mat = np.zeros((n_stim, n_stim))
        for srx in range(n_stim):
            for sry in range(n_stim):
                mean_val = np.mean([int(all_stim_responses[srx][x] == all_stim_responses[sry][x]) for x in range(3)])
                stim_conf_mat[srx, sry] = mean_val

        # save output stim x stim matrix
        sub_run_out_fpath = os.path.join(sub_beh_out_dir, 
                                         f'{sub_id}_{run_id}_stimulus_confusion_matrix.tsv')
        np.savetxt(sub_run_out_fpath, stim_conf_mat, delimiter='\t')