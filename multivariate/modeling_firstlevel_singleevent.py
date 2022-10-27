
import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn import plotting

''' Input parsing '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog='Example: python bids_modeling.py --sub=FLT02 --task=tonecat --space=T1w --fwhm=1.5 --event_type=sound --t_acq=2 --t_r=3'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--task", help="task id", type=str)
parser.add_argument("--space", help="space label", type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", type=float)
parser.add_argument("--event_type", help="what to model (options: `stimulus` or `feedback`)", type=str)
parser.add_argument("--t_acq", help="BOLD acquisition time (if different from repetition time [TR], as in sparse designs)", type=float)
parser.add_argument("--t_r", help="BOLD repetition time", type=float)

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
t_acq = args.t_acq
t_r = args.t_r


''' import data with `pybids` '''
# based on: https://github.com/bids-standard/pybids/blob/master/examples/pybids_tutorial.ipynb
def import_bids_data(bidsroot, subject_id, task_label):
    from bids import BIDSLayout

    layout = BIDSLayout(bidsroot)

    all_files = layout.get()
    t1w_fpath = layout.get(return_type='filename', subject=subject_id, 
                            suffix='T1w', extension='nii.gz')[0]
    bold_files = layout.get(return_type='filename', subject=subject_id, 
                            suffix='bold', task=task_label, extension='nii.gz')
    return all_files, t1w_fpath, bold_files


''' nilearn modeling: first level '''
def prep_models_and_args(subject_id=None, task_label=None, fwhm=None, bidsroot=None, 
                         deriv_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    data_dir = bidsroot

    task_label = task_label
    fwhm_sub = fwhm

    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r (which includes the silent gap),
    # we need to account for this
    slice_time_ref = 0.5 * t_acq / t_r

    print(data_dir, task_label, space_label)

    models, models_run_imgs, models_events, models_confounds = first_level_from_bids(data_dir, task_label, space_label,
                                                                                     [subject_id],
                                                                                     smoothing_fwhm=fwhm,
                                                                                     derivatives_folder=deriv_dir,
                                                                                     slice_time_ref=slice_time_ref)

    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = ['framewise_displacement',
                    'a_comp_cor_00', 'a_comp_cor_01', 
                    'a_comp_cor_02', 'a_comp_cor_03', 
                    'a_comp_cor_04', #'a_comp_cor_05', 
                    #'a_comp_cor_06', 'a_comp_cor_07', 
                    #'a_comp_cor_08', 'a_comp_cor_09', 
                    'trans_x', 'trans_y', 'trans_z', 
                    'rot_x','rot_y', 'rot_z']

    # create events
    if event_type == 'stimulus':
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

    # WIP
    elif event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):

                run_events['trial_type'][run_events['trial_type'].str[:2] == 'di'] = 'sound' 

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list


# Run-by-run GLM fit
def nilearn_glm_per_run(stim_list, task_label, event_filter, models, models_run_imgs, \
                        models_events, models_confounds, conf_keep_list, space_label):
    from nilearn.reporting import make_glm_report
    for midx in range(len(models)):
        stim_contrast_list = []
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim
            
            if event_filter in stim:
                print('running GLM with stimulus ', stim)

                midx = 0
                model = models[midx]
                imgs = models_run_imgs[midx]
                events = models_events[midx]
                confounds = models_confounds[midx]

                print(model.subject_label)

                # set limited confounds
                print('selecting confounds')
                confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]

                for rx in range(len(imgs)):
                    img = imgs[rx]
                    event = events[rx]
                    confound = confounds_ltd[rx]

                    try:
                        # fit the GLM
                        print('fitting GLM on ', img)
                        model.fit(img, event, confound);

                        # compute the contrast of interest
                        print('computing contrast of interest', ' with contrast label = ', contrast_label)
                        statmap = model.compute_contrast(contrast_label, output_type='effect_size')

                        # save z map
                        print('saving beta map')
                        nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                                    'level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                                    'sub-%s_space-%s'%(model.subject_label, space_label))
                        nilearn_sub_run_dir = os.path.join(nilearn_sub_dir, 'trial_models', 'run%02d'%rx)

                        if not os.path.exists(nilearn_sub_run_dir):
                            os.makedirs(nilearn_sub_run_dir)

                        analysis_prefix = ('sub-%s_task-%s_fwhm-%.02f_'
                                           'space-%s_contrast-%s_run%02d'%(model.subject_label,
                                                                           task_label, model.smoothing_fwhm,
                                                                           space_label, contrast_desc,
                                                                           rx))
                        statmap_fpath = os.path.join(nilearn_sub_run_dir,
                                                analysis_prefix+'_map-beta.nii.gz')
                        nib.save(statmap, statmap_fpath)
                        print('saved beta map to ', statmap_fpath)

                        stim_contrast_list.append(contrast_label)

                    except:
                        print('could not run for ', img, ' with ', contrast_label)
     
''' Run pipelines '''
project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
bidsroot = os.path.join(project_dir,'data_bids_noIntendedFor')
deriv_dir = os.path.join(project_dir, 'derivatives', 'fmriprep_noSDC')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
if not os.path.exists(nilearn_dir):
        os.makedirs(nilearn_dir)

''' Single-event modeling for multivariate analysis '''
if event_type == 'stimulus':
    event_filter = 'sound'
stim_list, models, models_run_imgs, \
    models_events, models_confounds, conf_keep_list = prep_models_and_args(subject_id, task_label, 1.5, bidsroot, 
                                                                           deriv_dir, event_type, t_r, t_acq, 
                                                                           space_label=space_label)
print('stim list: ', stim_list)
statmap_fpath, contrast_label = nilearn_glm_per_run(stim_list, task_label, event_filter, models, models_run_imgs, 
                                                 models_events, models_confounds, conf_keep_list, space_label='T1w')