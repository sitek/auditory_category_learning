
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn.glm.first_level import first_level_from_bids

''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python modeling_firstlevel_stimulus_perrun.py --sub=FLT02 '
                        '--task=tonecat --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound --model_type=LSS '
                        '--t_acq=2 --t_r=3 '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
                )

parser.add_argument("--sub", 
                    help="participant id", type=str)
parser.add_argument("--task", 
                    help="task id", type=str)
parser.add_argument("--space", 
                    help="space label", type=str)
parser.add_argument("--fwhm", 
                    help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--event_type", 
                    help="what to model (options: `trial`, `sound`, `stimulus`, `feedback`, or `motor`)", 
                    type=str)
parser.add_argument("--model_type", 
                    help="trial model scheme (options: `LSA` or `LSS`)", 
                    type=str)
parser.add_argument("--t_acq", 
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"), 
                    type=float)
parser.add_argument("--t_r", 
                    help="BOLD repetition time", 
                    type=float)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
subject_id = args.sub
task_label = args.task
space_label=args.space
fwhm = args.fwhm
event_type=args.event_type
model_type=args.model_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# correct the fmriprep-given slice reference (middle slice, or 0.5)
# to account for sparse acquisition (silent gap during auditory presentation paradigm)
# fmriprep is explicitly based on slice timings, while nilearn is based on t_r
# and since images are only collected during a portion of the overall t_r 
# (which includes the silent gap),
# we need to account for this
slice_time_ref = 0.5 * t_acq / t_r
    
''' ## nilearn modeling: first level '''
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html
# #sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

def update_events(models_events, event_type='sound'):
    ''' create events '''
    # stimulus events
    if event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])
    
    # trial-specific events
    if event_type == 'trial':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')
                
                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

    # all sound events
    if event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
                print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str.split('_', 
                                                                           expand=True)[0]

                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
        print('stim list: ', stim_list)
        
    # motor events by response type
    if event_type == 'motor':
        new_models_events = []
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if 'resp_' in str(s)])
                print('original stim list: ', orig_stim_list)

                run_events = run_events[run_events.trial_type.str.contains('resp_', regex=False)]
                                
                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

                #print(run_events)
                new_models_events.append(run_events)

        # create stimulus list from updated events.tsv file
        stim_list = orig_stim_list # sorted([str(s) for s in run_events['trial_type'].unique() if 'resp_' in str(s)])
        print('stim list: ', stim_list)
        
        # put new events into existing structure
        models_events = [new_models_events]

    return stim_list, models_events


# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,'trial_type'] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, 'trial_type'] = 'other_events' 
    return lss_event_df

# ### Run-by-run GLM fit
def nilearn_glm_per_run(stim_list, task_label,
                        event_filter,
                        models, models_run_imgs,
                        models_events, models_confounds,
                        space_label,
                        model_type='LSA'):
    from nilearn.interfaces.bids import save_glm_to_bids
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    
    midx = 0 # only 1 subject per analysis
    model = models[midx]

    # set limited confounds
    print('selecting confounds')
    imgs = models_run_imgs[midx]
    confounds_ltd, sample_mask = load_confounds_strategy(img_files=imgs, 
                                                         denoise_strategy='compcor')
    
    for contrast_label in stim_list:
        if event_filter not in contrast_label:
            continue
        else:
            print('running GLM with stimulus ', contrast_label)

        # for each run
        for rx, img in enumerate(imgs):
            confound = confounds_ltd[rx]

            if model_type == 'LSA':
                event = models_events[midx][rx]
            elif model_type == 'LSS':
                event = lss_transformer(models_events[midx][rx], contrast_label)

            #try:
            # fit the GLM
            print('fitting GLM on ', img)
            model.fit(img, event, confound);

            # save stat maps
            print('saving stat maps')

            from nilearn.interfaces.bids import save_glm_to_bids
            bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                             f'bids-deriv_level-1_fwhm-{model.smoothing_fwhm:.02f}', 
                                             f'sub-{model.subject_label}_space-{space_label}',
                                             f'per_run_{model_type}_confound-compcor_event-{event_type}', 
                                             f'run{rx:02}')
            if not os.path.exists(bidsderiv_sub_dir):
                os.makedirs(bidsderiv_sub_dir)

            out_prefix = f"sub-{model.subject_label}_run-{rx}_task-{task_label}_fwhm-{model.smoothing_fwhm}"
            save_glm_to_bids(model, 
                             contrast_label,
                             out_dir=bidsderiv_sub_dir,
                             prefix=out_prefix,
                            )
            print(f'Saved model outputs to {bidsderiv_sub_dir}')

            #except:
            #    print('could not run for ', img, ' with ', contrast_label)
          
''' Multivariate analysis: across-run GLM '''
print('running with subject ', subject_id)
#event_type = 'stimulus'
event_filter = 'sound' # 'sound'
        
models, models_run_imgs, \
        models_events, \
        models_confounds = first_level_from_bids(bidsroot, 
                                                 task_label, 
                                                 space_label,
                                                 [subject_id],
                                                 smoothing_fwhm=fwhm,
                                                 derivatives_folder=fmriprep_dir,
                                                 slice_time_ref=None, #slice_time_ref,
                                                 minimize_memory=False)

stim_list, models_events = update_events(models_events, 
                                         event_type=event_type)
print('stim_list:', stim_list)

nilearn_glm_per_run(stim_list, task_label, 
                    event_filter, 
                    models, models_run_imgs, 
                    models_events, models_confounds, 
                    space_label=space_label,
                    model_type=model_type)