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
                epilog=('Example: python univariate_analysis.py --sub=FLT02 '
                        '--task=stgrid --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound --t_acq=2 --t_r=4 '
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
                    help="what to model (options: `stimulus` or `feedback`)", 
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


# ### import data with `pybids` 
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



# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

def prep_models_and_args(subject_id=None, task_id=None, fwhm=None, bidsroot=None, 
                         fmriprep_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds

    data_dir = bidsroot

    task_label = task_id
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
                                                                                     derivatives_folder=fmriprep_dir,
                                                                                     slice_time_ref=slice_time_ref,
                                                                                     minimize_memory=False)

    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = ['framewise_displacement',
    #                #'a_comp_cor_00', 'a_comp_cor_01', 
    #                #'a_comp_cor_02', 'a_comp_cor_03', 
    #                #'a_comp_cor_04', 'a_comp_cor_05', 
    #                #'a_comp_cor_06', 'a_comp_cor_07', 
    #                #'a_comp_cor_08', 'a_comp_cor_09', 
                    'trans_x', 'trans_y', 'trans_z', 
                    'rot_x','rot_y', 'rot_z']

    # create stimulus list from events.tsv file
    stim_list = sorted([str(s) for s in models_events[0][0]['trial_type'].unique() if str(s) not in ['nan', 'None']])
    
    print('stim list: ', stim_list)
    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list

# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,'trial_type'] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, 'trial_type'] = 'other_events' 
    return lss_event_df


# ### Across-runs GLM
def nilearn_glm_across_runs(stim_list, task_label, model_type, \
                            models, models_run_imgs, \
                            models_events, models_confounds, \
                            conf_keep_list, space_label):
    from nilearn.reporting import make_glm_report
    for midx in range(len(models)):
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim


            midx = 0
            model = models[midx]
            imgs = models_run_imgs[midx]
            #events = models_events[midx]
            confounds = models_confounds[midx]
            if model_type = 'LSA':
                events = models_events[midx]
            elif model_type == 'LSS':
                events = [lss_transformer(models_events[midx][rx], stim) for rx in range(len(imgs))]
            
            print(model.subject_label)

            # set limited confounds
            print('selecting confounds')
            confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]
            
            #try:
            # fit the GLM
            print('fitting GLM')
            model.fit(imgs, events, confounds_ltd);

            # compute the contrast of interest
            print('computing contrast of interest')
            summary_statistics = model.compute_contrast(contrast_label, output_type='all')
            zmap = summary_statistics['z_score']
            tmap = summary_statistics['stat']
            statmap = summary_statistics['effect_size']

            # get the residuals

            # save z map
            print('saving z-map')
            nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                        'level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                        'sub-%s_space-%s'%(model.subject_label, space_label),
                                        'task-%s_run-all'%task_label)
            if not os.path.exists(nilearn_sub_dir):
                os.makedirs(nilearn_sub_dir)

            analysis_prefix = 'sub-%s_task-%s_fwhm-%.02f_space-%s_contrast-%s'%(model.subject_label,
                                                                                task_label, model.smoothing_fwhm,
                                                                                space_label, contrast_desc)
            zmap_fpath = os.path.join(nilearn_sub_dir,
                                    analysis_prefix+'_zmap.nii.gz')
            nib.save(zmap, zmap_fpath)
            print('saved z map to ', zmap_fpath)

            tmap_fpath = os.path.join(nilearn_sub_dir,
                                    analysis_prefix+'_map-tstat.nii.gz')
            nib.save(tmap, tmap_fpath)
            print('saved t map to ', tmap_fpath)

            # also save beta maps
            statmap_fpath = os.path.join(nilearn_sub_dir,
                                        analysis_prefix+'_map-beta.nii.gz')
            nib.save(statmap, statmap_fpath)
            print('saved beta map to ', statmap_fpath)

            # save report
            print('saving report')
            report_fpath = os.path.join(nilearn_sub_dir,
                                        analysis_prefix+'_report.html')
            report = make_glm_report(model=model,
                                    contrasts=contrast_label)
            report.save_as_html(report_fpath)
            print('saved report to ', report_fpath)
            #except:
            #    print('could not run for ', contrast_label)
    return zmap_fpath, statmap_fpath, contrast_label,

nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')
if not os.path.exists(nilearn_dir):
        os.makedirs(nilearn_dir)
        
event_type = 'block_stim'

stim_list, models, models_run_imgs, \
    models_events, models_confounds, \
    conf_keep_list = prep_models_and_args(subject_id, task_label, 
                                          fwhm, bidsroot, 
                                          fmriprep_dir, event_type,
                                          t_r, t_acq, 
                                          space_label)
# Across-run GLM
zmap_fpath, statmap_fpath, \
    contrast_label = nilearn_glm_across_runs(stim_list, task_label, model_type,
                                             models, models_run_imgs, 
                                             models_events, models_confounds, 
                                             conf_keep_list, space_label)
