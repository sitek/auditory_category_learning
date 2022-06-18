#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import numpy as np

import bids
from bids import BIDSLayout

from glob import glob

parser = argparse.ArgumentParser(
                description='Generate bids-compatible event file from psychopy log',
                epilog='Example: python bids_modeling.py --sub=FLT02 --task=tonecat --fwhm=1.5'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--task", help="task id", type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", type=float)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_id = args.task
fwhm = args.fwhm
#subject_id = 'FLT01'

project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
bidsroot = os.path.join(project_dir,'data_bids')
deriv_dir = os.path.join(project_dir, 'derivatives')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
if not os.path.exists(nilearn_dir):
        os.makedirs(nilearn_dir)


# #### define T1w-space aparc+aseg file (from fmriprep)

aparc_fpath = os.path.join(deriv_dir, 'fmriprep/',
                           'sub-%s/anat'%subject_id,
                           'sub-%s_desc-aparcaseg_dseg.nii.gz'%subject_id)

# ### import data with `pybids` and test out capabilities
# based on: https://github.com/bids-standard/pybids/blob/master/examples/pybids_tutorial.ipynb

layout = BIDSLayout(bidsroot)

all_files = layout.get()
t1w_fpath = layout.get(return_type='filename', suffix='T1w', extension='nii.gz')[0]
bold_files = layout.get(return_type='filename', subject=subject_id, 
                        suffix='bold', task=task_id,extension='nii.gz')


# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py
def prep_models_and_args(subject_id, task_id, fwhm, bidsroot, deriv_dir, space_label='T1w', event_type):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.reporting import make_glm_report

    import nibabel as nib

    from nilearn import plotting
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    data_dir = bidsroot
    derivatives_folder = os.path.join(deriv_dir, 'fmriprep')


    task_label = task_id
    fwhm_sub = fwhm # 1.5

    print(data_dir, task_label, space_label)
    models, models_run_imgs, models_events, models_confounds = first_level_from_bids(data_dir, task_label, space_label,
                                                                                    smoothing_fwhm=fwhm_sub,
                                                                                    derivatives_folder=derivatives_folder,
                                                                                    slice_time_ref=0.5)

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

    return models_and_args, stim_list

# ### Across-runs GLM
def nilearn_glm_across_runs(model_and_args, stim_list):
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
    return zmap_fpath, contrast_label

def plot_stat_maps():
    from scipy.stats import norm
    p_val = 0.005
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
def plot_tsnr():
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

# #### generate anatomical STG masks
def create_cortical_masks():
    from scipy.ndimage import binary_dilation
    from nilearn.image import resample_to_img

    print(aparc_fpath)
    aparc_img = nib.load(aparc_fpath)
    aparc_data = aparc_img.get_fdata()
    aparc_affine = aparc_img.affine


    roi_dict = {'lh_stg': 1030, 'lh_hg': 1034, 'rh_stg': 2030, 'rh_hg': 2034}
    lh_labels = list(roi_dict.values())[:2]
    rh_labels = list(roi_dict.values())[2:]

    lh_mask = np.zeros(aparc_data.shape)
    lh_mask[np.where(aparc_data == 1030)] = 1
    lh_mask[np.where(aparc_data == 1034)] = 1


    rh_mask = np.zeros(aparc_data.shape)
    rh_mask[np.where(aparc_data == 2030)] = 1
    rh_mask[np.where(aparc_data == 2034)] = 1


    lh_mask_img = nib.Nifti1Image(lh_mask, aparc_affine)
    rh_mask_img = nib.Nifti1Image(rh_mask, aparc_affine)


    lh_mask_func_img = resample_to_img(lh_mask_img, zmap, interpolation='nearest')
    rh_mask_func_img = resample_to_img(rh_mask_img, zmap, interpolation='nearest')


    nib.save(lh_mask_func_img, os.path.join(nilearn_dir,
                                            'sub-%s_mask-L-aud-ctx.nii.gz'%model.subject_label))
    nib.save(rh_mask_func_img, os.path.join(nilearn_dir,
                                            'sub-%s_mask-R-aud-ctx.nii.gz'%model.subject_label))

    aud_mask = np.zeros(aparc_data.shape)
    aud_mask[np.where(lh_mask == 1)] = 1
    aud_mask[np.where(rh_mask == 1)] = 1

    aud_mask_dil = binary_dilation(aud_mask).astype(aud_mask.dtype)

    aud_mask_anat_img = nib.Nifti1Image(aud_mask_dil, affine=aparc_affine)
    aud_mask_func_img = resample_to_img(aud_mask_anat_img, zmap, interpolation='nearest')


    plotting.plot_stat_map(aud_mask_func_img, bg_img=t1w_fpath, colorbar=False,
                        display_mode='y', cut_coords=6);

    aud_mask_fpath = os.path.join(nilearn_dir,'sub-%s_mask-aud-ctx.nii.gz'%model.subject_label)
    nib.save(aud_mask_func_img, aud_mask_fpath)
    
    return aud_mask_fpath

# #### mask auditory regions
def mask_z_map_imgs():
    from nilearn.masking import apply_mask
    from nilearn.masking import unmask

    masked_data = apply_mask(zmap, aud_mask_func_img)
    masked_img = unmask(masked_data, aud_mask_func_img)

    p_val = 0.005
    thresh_unc = norm.isf(p_val)

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    plotting.plot_stat_map(masked_img, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        title='sub-%s %s (unc p<%.03f; fwhm=%.02f; STG mask)'%(model.subject_label, 
                                                                    task_label,p_val,fwhm_sub),
                        axes=axes[0],
                        display_mode='x', cut_coords=6)
    plotting.plot_stat_map(masked_img, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        axes=axes[1],
                        display_mode='y', cut_coords=6)
    plotting.plot_stat_map(masked_img, bg_img=t1w_fpath, colorbar=True, threshold=thresh_unc,
                        axes=axes[2],
                        display_mode='z', cut_coords=6)
    plotting.show()

    # save plot
    plot_fpath = os.path.join(nilearn_dir, 
                            'sub-%s_task-%s_fwhm-%.02f_pval-%.03f_space-%s_mask-aud.png'%(model.subject_label,
                                                                        task_label,fwhm_sub, p_val,
                                                                        space_label))
    fig.savefig(plot_fpath)
    return plot_fpath


# #### Load subcortical (MNI space) regions
def load_IC_MNI():
    subcort_atlas_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/',
                                    'data/reference/',
                                    'MNI_space/atlases/',
                                    'sub-invivo_MNI_rois.nii.gz')
    subcort_img = nib.load(subcort_atlas_fpath)

    subcort_func_img = resample_to_img(subcort_img, zmap, interpolation='nearest')

    mask_IC = np.zeros(zmap.shape)
    mask_IC[np.where(subcort_func_img.get_fdata() == 5)] = 1
    mask_IC[np.where(subcort_func_img.get_fdata() == 6)] = 1

    mask_IC_img = nib.Nifti1Image(mask_IC, affine = zmap.affine)

    mask_IC_fpath = os.path.join(nilearn_dir,
                                            'sub-%s_space-%s_mask-IC.nii.gz'%(model.subject_label, 
                                                                            space_label))
    nib.save(mask_IC_img, mask_IC_fpath)
    
    return mask_IC_fpath


# ## Decoding
def generate_conditions(sub_id, fwhm_sub, space_label, derivatives_dir):
    sub_id = 'FLT01'
    fwhm_sub = 1.5
    space_label = 'T1w' #'MNI152NLin2009cAsym' 

    nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                                'level-1_fwhm-%.02f'%fwhm_sub, 
                                                'sub-%s_space-%s'%(sub_id, space_label))
    print(nilearn_sub_dir)

    z_maps = sorted(glob(nilearn_sub_dir+'/trial_models'+'/run*/*di*nii.gz'))
    print('# of z-maps: ', len(z_maps))

    # 16 stimulus decoding
    conditions_all = [os.path.basename(x)[-31:-18] for x in z_maps] 
    print(conditions_all[:10])

    # 4-category decoding
    conditions_tone = [stim[:3] for stim in conditions_all]
    print(conditions_tone[:10])

    conditions_talker = [stim[4:6] for stim in conditions_all]
    print(conditions_talker[:10])

    # pick the labels
    conditions = conditions_tone

    print('# of trials: ', len(conditions))
    print(conditions[:10])
    print('unique # of conditions = ', np.unique(conditions).shape)
    # ### Nilearn `Decoder` accuracies
    # from https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_glm_decoding.html#build-the-decoding-pipeline
    # **note: does not generate predictions for each fold, so cannot use for confusion matrix generation**
    
    return z_maps, conditions

def region_decoding(z_maps, conditions, mask_descrip, n_runs):
    from nilearn.decoding import Decoder
    n_runs = 4
    split_index = round(len(z_maps) * (n_runs-1) / n_runs)
    print('# of training images = ', split_index)

    # available space-T1w masks: aud_mask_func_img, lh_mask_func_img, rh_mask_func_img
    # available space-MNI masks: mask_IC_img
    mask_descrip = 'L-TTG' #'aud-ctx'
    mask_fpath = nib.load(os.path.join(nilearn_dir, 'sub-%s_mask-%s.nii.gz'%(sub_id, mask_descrip)))
    #mask = mask_IC_fpath

    cv = 5
    decoder=Decoder(estimator='svc', mask=mask_fpath,
                    standardize=False,
                    screening_percentile=10, 
                    cv=5,
                )
    decoder.fit(z_maps[:split_index], conditions[:split_index])

    y_pred = decoder.predict(z_maps[split_index:])
    print(y_pred)

    classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
    chance_level = 1. / len(np.unique(conditions))
    print('{} classification accuracy: {:4f} / Chance level: {}'.format(
        sub_id, classification_accuracy, chance_level))

    # Create and save prediction accuracy plot
    masked_data_dir = os.path.join(nilearn_sub_dir, 'trial_models', 'masked_data')
    os.makedirs(masked_data_dir, exist_ok=True)

    region_string = mask_descrip

    acc_plot_fpath = os.path.join(masked_data_dir,
                                    '%s_space-%s_roi-%s_trial_decoding_accuracy.png'%(sub_id, 
                                                                            space_label,
                                                                            mask_descrip))

    from matplotlib import pyplot as plt
    #plt.figure(figsize=(8,3), dpi=150)
    plt.figure(figsize=(5,3), dpi=150)
    plt.boxplot(list(decoder.cv_scores_.values()));
    plt.axhline(y=chance_level, color='r', linewidth=0.5)
    plt.title('{} {} SVC accuracy: {:.03f} (Chance level: {})'.format(
                sub_id, region_string, classification_accuracy, chance_level))
    plt.xlabel('stimulus')
    plt.ylabel('accuracy')
    plt.ylim([0, 1])
    plt.xticks(range(1, len(np.unique(conditions))+1), 
            np.unique(conditions), 
            rotation=30);

    # save figure
    plt.savefig(acc_plot_fpath)
    print('saved figure to ', acc_plot_fpath)

    # save 4-D decoding coefficient images to nifti (3-D x trial type prediction)
    decoding_dir = os.path.join(masked_data_dir, 'decoding')
    os.makedirs(decoding_dir, exist_ok=True)
    print('saving files to ', decoding_dir)

    for ix, decoder_cond in enumerate(decoder.coef_img_):
        decoder_img = decoder.coef_img_[decoder_cond]
        decoder_img_fpath = os.path.join(decoding_dir,
                                        '%s_space-%s_roi-%s_trial_decoding_cond-%s.nii.gz'%(models[sx].subject_label, 
                                                                                space_label,
                                                                                mask_descrip,
                                                                                decoder_cond))
        nib.save(decoder_img, decoder_img_fpath)

    return decoder_img_fpath_list

# #### Extract and save matrix
def save_masked_conditions_timeseries(mask, z_maps, out_dir):
    from nilearn.input_data import NiftiMasker

    mask_descrip = 'L-TTG'
    mask_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/nilearn/',
                            'sub-%s_mask-%s.nii.gz'%(model.subject_label, mask_descrip))

    masker = NiftiMasker(mask_img=mask_fpath, smoothing_fwhm=None, standardize=False)
    masked_data = masker.fit_transform(z_maps)

    print(masked_data.shape)

    masked_data_fpath = os.path.join(nilearn_sub_dir, 'trial_models', 'masked_data',
                                    'sub-%s_space-%s_roi-%s_trial_zmaps.csv'%(model.subject_label, 
                                                                        space_label,
                                                                        mask_descrip))
    np.savetxt(masked_data_fpath, masked_data)

    conditions_fpath = os.path.join(nilearn_sub_dir, 'trial_models', 'masked_data',
                                    'sub-%s_space-%s_roi-%s_trial_conditions.csv'%(model.subject_label, 
                                                                            space_label,
                                                                            mask_descrip))
    np.savetxt(conditions_fpath, conditions, fmt='%s')

    return masked_data_fpath, conditions_fpath

# ### striatum
def create_mask_striatum(model, aparc_fpath, t1w_fpath):
    import nilearn.decoding
    from scipy.ndimage import binary_dilation, binary_erosion
    from nilearn.image import resample_to_img

    print(aparc_fpath)
    aparc_img = nib.load(aparc_fpath)
    aparc_data = aparc_img.get_fdata()
    aparc_affine = aparc_img.affine
    print(aparc_affine)

    t1w_img = nib.load(t1w_fpath)
    t1w_affine=t1w_img.affine
    print(t1w_affine)

    # #### define striatum masks

    lh_mask = np.zeros(aparc_data.shape)
    lh_mask[np.where(aparc_data == 11)] = 1
    lh_mask[np.where(aparc_data == 12)] = 1
    lh_mask[np.where(aparc_data == 13)] = 1
    lh_mask[np.where(aparc_data == 26)] = 1

    rh_mask = np.zeros(aparc_data.shape)
    rh_mask[np.where(aparc_data == 50)] = 1
    rh_mask[np.where(aparc_data == 51)] = 1
    rh_mask[np.where(aparc_data == 52)] = 1
    rh_mask[np.where(aparc_data == 58)] = 1

    lh_mask_img = nib.Nifti1Image(lh_mask, aparc_affine)
    rh_mask_img = nib.Nifti1Image(rh_mask, aparc_affine)

    striatum_mask = np.zeros(aparc_data.shape)
    striatum_mask[np.where(lh_mask == 1)] = 1
    striatum_mask[np.where(rh_mask == 1)] = 1

    striatum_mask_ero = binary_erosion(striatum_mask).astype(striatum_mask.dtype)
    striatum_mask_dil = binary_dilation(striatum_mask).astype(striatum_mask.dtype)

    striatum_mask_erodil = binary_dilation(striatum_mask_ero).astype(striatum_mask_ero.dtype)


    striatum_mask_fs_img = nib.Nifti1Image(striatum_mask, affine=aparc_affine)
    striatum_mask_ero_fs_img = nib.Nifti1Image(striatum_mask_ero, affine=aparc_affine)
    striatum_mask_dil_fs_img = nib.Nifti1Image(striatum_mask_dil, affine=aparc_affine)

    striatum_mask_erodil_fs_img = nib.Nifti1Image(striatum_mask_erodil, affine=aparc_affine)


    striatum_mask_anat_img = resample_to_img(striatum_mask_erodil_fs_img, t1w_fpath, 
                                            interpolation='nearest')
    striatum_mask_func_img = resample_to_img(striatum_mask_anat_img, zmap, 
                                            interpolation='nearest')
    striatum_mask_fpath = os.path.abspath('sub-%s_mask-striatum.nii.gz'%model.subject_label)

    nib.save(striatum_mask_anat_img,)
    return striatum_mask_fpath
