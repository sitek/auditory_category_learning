
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn import plotting

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
                    help="what to model (options: `sound` or `stimulus` or `feedback`)", 
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

''' ## nilearn modeling: first level '''
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html
# #sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

def prep_models_and_args(subject_id=None, task_label=None, fwhm=None, bidsroot=None, 
                         deriv_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r 
    # (which includes the silent gap),
    # we need to account for this
    slice_time_ref = 0.5 * t_acq / t_r

    print(bidsroot, task_label, space_label)

    models, models_run_imgs, \
            models_events, \
            models_confounds = first_level_from_bids(bidsroot, 
                                                     task_label, 
                                                     space_label,
                                                     [subject_id],
                                                     smoothing_fwhm=fwhm,
                                                     derivatives_folder=deriv_dir,
                                                     slice_time_ref=slice_time_ref,
                                                     minimize_memory=False)

    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = ['framewise_displacement',
                    #'a_comp_cor_00', 'a_comp_cor_01', 
                    #'a_comp_cor_02', 'a_comp_cor_03', 
                    #'a_comp_cor_04', 'a_comp_cor_05', 
                    #'a_comp_cor_06', 'a_comp_cor_07', 
                    #'a_comp_cor_08', 'a_comp_cor_09', 
                    'trans_x', 'trans_y', 'trans_z', 
                    'rot_x','rot_y', 'rot_z']

    ''' create events '''
    # stimulus events
    if event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])
    
    # trial-specific events
    if event_type == 'trial':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

    # all sound events
    elif event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
                print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str.split('_', 
                                                                           expand=True)[0]

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
        print('stim list: ', stim_list)

    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list


# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,'trial_type'] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, 'trial_type'] = 'other_events' 
    return lss_event_df

# ### Run-by-run GLM fit
def nilearn_glm_per_run(stim_list, task_label, \
                        event_filter, model_type, \
                        models, models_run_imgs, \
                        models_events, models_confounds, \
                        conf_keep_list, space_label):
    from nilearn.reporting import make_glm_report
    
    # for each model (corresponding to a subject)
    for midx in range(len(models)):
        stim_contrast_list = []
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim
            
            #if event_filter in stim: run on only sound events
            #if event_filter not in stim: # run on non-sound events
            print('running GLM with stimulus ', stim)

            model = models[midx]

            print(model.subject_label)

            # set limited confounds
            print('selecting confounds')
            confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]

            # for each run
            for rx in range(len(confounds_ltd)):
                img = models_run_imgs[midx][rx]
                confound = confounds_ltd[rx]

                if model_type == 'LSA':
                    event = models_events[midx][rx]
                elif model_type == 'LSS':
                    event = lss_transformer(models_events[midx][rx], stim)
                print(sorted(event.trial_type.unique()))

                try:
                    # fit the GLM
                    print('fitting GLM on ', img)
                    model.fit(img, event, confound);

                    # compute the contrast of interest
                    print('computing contrast of interest', 
                          ' with contrast label = ', contrast_label)
                    summary_statistics = model.compute_contrast(contrast_label, 
                                                                output_type='all')
                    zmap = summary_statistics['z_score']
                    tmap = summary_statistics['stat']
                    statmap = summary_statistics['effect_size']
                    varmap = summary_statistics['effect_variance']

                    # save stat maps
                    print('saving stat maps')
                    nilearn_sub_dir = os.path.join(bidsroot, 
                                                   'derivatives', 
                                                   'nilearn', 
                                                'level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                                'sub-%s_space-%s'%(model.subject_label, 
                                                                   space_label))
                    nilearn_sub_run_dir = os.path.join(nilearn_sub_dir, 
                                                       'stimulus_per_run_%s'%model_type, 
                                                       'run%02d'%rx)

                    if not os.path.exists(nilearn_sub_run_dir):
                        os.makedirs(nilearn_sub_run_dir)

                    analysis_prefix = ('sub-%s_task-%s_fwhm-%.02f_'
                                       'space-%s_contrast-%s_run%02d_'
                                       'model-%s'%(model.subject_label,
                                                   task_label, model.smoothing_fwhm,
                                                   space_label, contrast_desc,
                                                   rx, model_type))
                    statmap_fpath = os.path.join(nilearn_sub_run_dir,
                                            analysis_prefix+'_map-beta.nii.gz')

                    nib.save(statmap, statmap_fpath)
                    print('saved beta map to ', statmap_fpath)

                    # save t map
                    tmap_fpath = os.path.join(nilearn_sub_run_dir,
                                            analysis_prefix+'_map-tstat.nii.gz')
                    nib.save(tmap, tmap_fpath)
                    print('saved t map to ', tmap_fpath)

                    # save var map
                    varmap_fpath = os.path.join(nilearn_sub_run_dir,
                                            analysis_prefix+'_map-var.nii.gz')
                    nib.save(varmap, varmap_fpath)
                    print('saved var map to ', varmap_fpath)

                    '''
                    # save residuals
                    resid_fpath = os.path.join(nilearn_sub_run_dir,
                                            analysis_prefix+'_map-residuals.nii.gz')
                    nib.save(model.residuals[0], resid_fpath)
                    print('saved residuals map to ', resid_fpath)
                    '''

                    # save report
                    print('saving report')
                    report_fpath = os.path.join(nilearn_sub_run_dir,
                                                analysis_prefix+'_report.html')
                    report = make_glm_report(model=model,
                                            contrasts=contrast_label)
                    report.save_as_html(report_fpath)
                    print('saved report to ', report_fpath)

                except:
                    print('could not run for ', img, ' with ', contrast_label)
          
''' Multivariate analysis: across-run GLM '''
print('running with subject ', subject_id)
event_type = 'stimulus'
event_filter = 'sound'
        
stim_list, models, models_run_imgs, \
    models_events, models_confounds, \
    conf_keep_list = prep_models_and_args(subject_id, task_label, fwhm, bidsroot, 
                                          fmriprep_dir, event_type, t_r, t_acq, 
                                          space_label=space_label)
print('stim list: ', stim_list)
statmap_fpath, contrast_label = nilearn_glm_per_run(stim_list, task_label, 
                                                    event_filter, model_type,
                                                    models, models_run_imgs, 
                                                    models_events, models_confounds, 
                                                    conf_keep_list, space_label=space_label)