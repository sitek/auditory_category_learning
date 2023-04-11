
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
                epilog=('Example: python modeling_firstlevel_singleevent.py --sub=FLT02 '
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

''' nilearn modeling: first level '''
def prep_models_and_args(subject_id=None, task_label=None, fwhm=None, bidsroot=None, 
                         deriv_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids

    task_label = task_label
    fwhm_sub = fwhm

    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r (which includes the silent gap),
    # we need to account for this
    slice_time_ref = 0.5 * t_acq / t_r

    print(bidsroot, task_label, space_label)

    models, models_run_imgs, \
            models_events, models_confounds = first_level_from_bids(bidsroot, 
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

    # create events
    if event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type']
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])
        
    #model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list

# from nilearn docs 
# https://nilearn.github.io/stable/auto_examples/07_advanced/plot_beta_series.html#define-the-lss-models
def lss_transformer(df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the DataFrame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information, with the select trial's trial type isolated.
    trial_name : str
        Name of the isolated trial's trial type.
    """
    df = df.copy()

    # Determine which number trial it is *within the condition*
    trial_condition = df.loc[row_number, 'trial_type']
    trial_type_series = df['trial_type']
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the dataframe, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that you could track individual trials across models.
    trial_name = f'{trial_condition}_trial{trial_number:02d}'
    df.loc[row_number, 'trial_type'] = trial_name
    return df, trial_name

# different version based on
# https://nibetaseries.readthedocs.io/en/stable/betaseries.html#mathematical-background
def lss_transformer(event_df, row_number):
    '''
    di1_idxs = np.where(event.trial_type.str.contains('di1'))
    di2_idxs = np.where(event.trial_type.str.contains('di2'))
    di3_idxs = np.where(event.trial_type.str.contains('di3'))
    di4_idxs = np.where(event.trial_type.str.contains('di4'))
    resp_idxs = np.where(event.trial_type.str.contains('resp'))
    fb_idxs = np.where(event.trial_type.str.contains('fb'))
    
    if 'di1' in event.iloc[row_number].trial_type:
        idxs = 
    '''
    

# Run-by-run GLM fit
def nilearn_glm_per_run(stim_list, task_label, event_filter, models, models_run_imgs, \
                        models_events, models_confounds, conf_keep_list, space_label):
    from nilearn.reporting import make_glm_report
    for midx in range(len(models)):
        model = models[midx]

        print(model.subject_label)

        # set limited confounds
        print('selecting confounds')
        confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]

        for rx in range(len(imgs)):
            img = models_run_imgs[midx][rx]
            event = models_events[midx][rx]
            confound = confounds_ltd[rx]

            for i_trial in range(event.shape[0]):
                if event_filter in stim:
                    print('running GLM with stimulus ', stim)
                    lss_event, cond_name = lss_transformer(event, i_trial)
                    contrast_label = cond_name
                    contrast_desc  = cond_name
                    print('LSS condition = ', cond_name)

                    # fit the GLM
                    print('fitting GLM on ', img)
                    model.fit(img, lss_event, confound);

                    # compute the contrast of interest
                    print('computing contrast of interest', ' with contrast label = ', contrast_label)
                    summary_statistics = model.compute_contrast(contrast_label, output_type='all')
                    #zmap = summary_statistics['z_score']
                    tmap = summary_statistics['stat']
                    statmap = summary_statistics['effect_size']

                    # save z map
                    print('saving beta map')
                    nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                                'level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                                'sub-%s_space-%s'%(model.subject_label, space_label))
                    nilearn_sub_run_dir = os.path.join(nilearn_sub_dir, 
                                                       'trial_models_%s'%model_type, 
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

                    #except:
                    #    print('could not run for ', img, ' with ', contrast_label)


''' Single-event modeling for multivariate analysis '''
if event_type == 'stimulus':
    event_filter = 'sound'
stim_list, models, models_run_imgs, \
    models_events, models_confounds, conf_keep_list = prep_models_and_args(subject_id, task_label, fwhm, bidsroot, 
                                                                           fmriprep_dir, event_type, t_r, t_acq, 
                                                                           space_label=space_label)
print('stim list: ', stim_list)
statmap_fpath, contrast_label = nilearn_glm_per_run(stim_list, task_label, event_filter, 
                                                    models, models_run_imgs, 
                                                    models_events, models_confounds, 
                                                    conf_keep_list, space_label=space_label)
