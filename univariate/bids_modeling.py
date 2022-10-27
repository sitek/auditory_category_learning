#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn import plotting

parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog='Example: python bids_modeling.py --sub=FLT02 --task=tonecat --space=T1w --fwhm=1.5 --event_type=stimulus'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--task", help="task id", type=str)
parser.add_argument("--space", help="space label", type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", type=float)
parser.add_argument("--event_type", help="what to model (options: `stimulus` or `feedback`)", type=float)
parser.add_argument("--t_acq", help="BOLD acquisition time (if different from repetition time [TR], as in sparse designs", type=float)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_id = args.task
space_label=args.space
fwhm = args.fwhm
event_type=args.event_type
t_acq = args.t_acq

project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
#bidsroot = os.path.join(project_dir,'data_bids')
bidsroot = os.path.join(project_dir,'data_bids_TEMP')
deriv_dir = os.path.join(project_dir, 'derivatives')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
if not os.path.exists(nilearn_dir):
        os.makedirs(nilearn_dir)

# ### import data with `pybids` 
# based on: https://github.com/bids-standard/pybids/blob/master/examples/pybids_tutorial.ipynb
def import_bids_data(bidsroot):
    from bids import BIDSLayout

    layout = BIDSLayout(bidsroot)

    all_files = layout.get()
    t1w_fpath = layout.get(return_type='filename', suffix='T1w', extension='nii.gz')[0]
    bold_files = layout.get(return_type='filename', subject=subject_id, 
                            suffix='bold', task=task_id,extension='nii.gz')
    return all_files, t1w_fpath, bold_files

# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py
def prep_models_and_args(subject_id, task_id, fwhm, bidsroot, deriv_dir, event_type, t_r, t_acq, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    data_dir = bidsroot
    derivatives_folder = os.path.join(deriv_dir, 'fmriprep')

    task_label = task_id
    fwhm_sub = fwhm # 1.5
    
    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r (which includes the silent gap),
    # we need to account for this
    slice_time_ref = 0.5 * t_acq / t_r

    print(data_dir, task_label, space_label)
    
    models, models_run_imgs, models_events, models_confounds = first_level_from_bids(data_dir, task_label, space_label,
                                                                                    smoothing_fwhm=fwhm_sub,
                                                                                    derivatives_folder=derivatives_folder,
                                                                                    slice_time_ref=slice_time_ref)
    
    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = ['framewise_displacement',
                    'a_comp_cor_00', 'a_comp_cor_01', 
                    'a_comp_cor_02', 'a_comp_cor_03', 
                    'a_comp_cor_04', 'a_comp_cor_05', 
                    'a_comp_cor_06', 'a_comp_cor_07', 
                    'a_comp_cor_08', 'a_comp_cor_09', 
                    'trans_x', 'trans_y', 'trans_z', 
                    'rot_x','rot_y', 'rot_z']

    # create events
    if event_type == 'feedback':
        stim_list = sorted([s for s in models_events[0][0]['feedback'].unique() if str(s) != 'nan'])
        
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['feedback']
                run_events.loc[run_events.duration != 0.3, 'trial_type'] = np.nan
                print(run_events['trial_type'])
        
        stim_list = sorted([s for s in models_events[1][1]['trial_type'].unique() if str(s) != 'nan'])

        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                #run_events['trial_type'] = run_events['trial_type'] + '_trial' + \
                #                                     np.arange(len(run_events)).astype(str)

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')
                print(run_events['trial_type'])
        

        stim_list = []
        stim_list.extend(['right_' + str(x) for x in range(0,61)])
        stim_list.extend(['wrong_' + str(x) for x in range(0,61)])
    
    elif event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            print(models[sx].subject_label)
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                #run_events['trial_type'] = run_events['trial_type'] + '_trial' + \
                #                                     np.arange(len(run_events)).astype(str)

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')
                print(run_events['trial_type'])

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

        
    model_and_args = zip(models, models_run_imgs, models_events, models_confounds)

    return model_and_args, stim_list

# ### Across-runs GLM
def nilearn_glm_across_runs(model_and_args, stim_list):
    from nilearn.reporting import make_glm_report
    for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim


            midx = 0
            model = models[midx]
            imgs = models_run_imgs[midx]
            events = models_events[midx]
            confounds = models_confounds[midx]

            print(model.subject_label)

            # set limited confounds
            print('selecting confounds')
            confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]

            # fit the GLM
            print('fitting GLM')
            model.fit(imgs, events, confounds_ltd);

            # compute the contrast of interest
            print('computing contrast of interest')
            zmap = model.compute_contrast(contrast_label)

            # save z map
            print('saving z-map')
            nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                        'level-1_fwhm-%.02f'%fwhm_sub, 
                                        'sub-%s_space-%s'%(model.subject_label, space_label),
                                        'run-all')
            if not os.path.exists(nilearn_sub_dir):
                os.makedirs(nilearn_sub_dir)

            analysis_prefix = 'sub-%s_task-%s_fwhm-%.02f_space-%s_contrast-%s'%(model.subject_label,
                                                                                task_label, fwhm_sub,
                                                                                space_label, contrast_desc)
            zmap_fpath = os.path.join(nilearn_sub_dir,
                                    analysis_prefix+'_zmap.nii.gz')
            nib.save(zmap, zmap_fpath)
            print('saved z map to ', zmap_fpath)

            # save report
            print('saving report')
            report_fpath = os.path.join(nilearn_sub_dir,
                                        analysis_prefix+'_report.html')
            report = make_glm_report(model=model,
                                    contrasts=contrast_label)
            report.save_as_html(report_fpath)
            print('saved report to ', report_fpath)


# ### Run-by-run GLM fit
def nilearn_glm_per_run(model_and_args, stim_list):
    from nilearn.reporting import make_glm_report
    for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
        stim_contrast_list = []
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim
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
                    zmap = model.compute_contrast(contrast_label)

                    # save z map
                    print('saving z-map')
                    nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                                'level-1_fwhm-%.02f'%fwhm_sub, 
                                                'sub-%s_space-%s'%(model.subject_label, space_label))
                    nilearn_sub_run_dir = os.path.join(nilearn_sub_dir, 'trial_models', 'run%02d'%rx)

                    if not os.path.exists(nilearn_sub_run_dir):
                        os.makedirs(nilearn_sub_run_dir)

                    analysis_prefix = ('sub-%s_task-%s_fwhm-%.02f_'
                                    'space-%s_contrast-%s_run%02d'%(model.subject_label,
                                                                    task_label, fwhm_sub,
                                                                    space_label, contrast_desc,
                                                                    rx))
                    zmap_fpath = os.path.join(nilearn_sub_run_dir,
                                            analysis_prefix+'_zmap.nii.gz')
                    nib.save(zmap, zmap_fpath)
                    print('saved z map to ', zmap_fpath)
                    
                    stim_contrast_list.append(contrast_label)

                    # save report
                    print('saving report')
                    report_fpath = os.path.join(nilearn_sub_dir,
                                                analysis_prefix+'_report.html')
                    report = make_glm_report(model=model,
                                            contrasts=contrast_label)
                    report.save_as_html(report_fpath)
                    print('saved report to ', report_fpath)
                except:
                    print('could not run for ', img, ' with ', contrast_label)

def plot_stat_maps(zmap, nilearn_dir, p_val=0.005):
    from scipy.stats import norm
    thresh_unc = norm.isf(p_val)

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    plotting.plot_stat_map(zmap, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        title='sub-%s %s (unc p<%.03f; fwhm=%.02f)'%(model.subject_label, 
                                                                    contrast_label ,p_val,fwhm_sub),
                        axes=axes[0],
                        display_mode='x', cut_coords=6)
    plotting.plot_stat_map(zmap, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        axes=axes[1],
                        display_mode='y', cut_coords=6)
    plotting.plot_stat_map(zmap, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        axes=axes[2],
                        display_mode='z', cut_coords=6)
    plotting.show()

    # save plot
    plot_fpath = os.path.join(nilearn_dir, 
                            'sub-%s_task-%s_fwhm-%.02f_pval-%.03f_space-%s_contrast-%s.png'%(model.subject_label,
                                                                        task_label,fwhm_sub, p_val,
                                                                        space_label, contrast_desc))
    fig.savefig(plot_fpath)

    return plot_fpath

# ## plot tsnr
def plot_tsnr(bold_files):
    from nilearn import image

    thresh = 0
    fwhm = 5

    for fx,filepath in enumerate(bold_files):
        tsnr_func = image.math_img('img.mean(axis=3) / img.std(axis=3)', img=filepath)
        tsnr_func_smooth = image.smooth_img(tsnr_func, fwhm=5)

        display = plotting.plot_stat_map(tsnr_func_smooth, 
                                        bg_img=t1w_fpath, 
                                        #title='fMRI single run tSNR map',
                                        #cut_coords=[8,50,-20],
                                        #threshold=thresh, 
                                        #cmap='jet'
                                        );
        display.show()

# transform MNI-space atlas into subject's T1w space
def transform_atlas_mni_to_t1w(t1w_fpath, atlas_fpath, transform_fpath):
    from ants import apply_transforms
    
    t1w_fpath = '/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep/' + \
                      'sub-FLT01/anat/sub-FLT01_desc-preproc_T1w.nii.gz'
    transform_fpath = '/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep/' + \
                      'sub-FLT01/anat/sub-FLT01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'

    atlas_spacet1w_fpath = apply_transforms(t1w_fpath, atlas_fpath, [transform_fpath], interpolator='multiLabel')

    return atlas_spacet1w_fpath

# run pipeline

all_files, t1w_fpath, bold_files = import_bids_data(bidsroot)

model_and_args, stim_list = prep_models_and_args(subject_id, task_id, fwhm, bidsroot, 
                                                 deriv_dir, event_type, t_r, t_acq, space_label)
zmap_fpath, contrast_label = nilearn_glm_per_run(model_and_args, stim_list)
z_maps, conditions = generate_conditions(subject_id, fwhm, space_label, deriv_dir)
decoder_img_fpath_list = region_decoding(subject_id, space_label, z_maps, conditions, mask_descrip, n_runs)
masked_data_fpath, conditions_fpath = save_masked_conditions_timeseries(mask, z_maps, out_dir)

