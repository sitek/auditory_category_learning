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
                epilog='Example: python bids_modeling.py --sub=FLT02 --task=tonecat'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--task", help="task id", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id = args.sub
task_id = args.task
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
def nilearn_first_level(subject_id, task_id, bidsroot, deriv_dir,
                        space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.reporting import make_glm_report

    import nibabel as nib

    from nilearn import plotting
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    data_dir = bidsroot
    derivatives_folder = os.path.join(deriv_dir, 'fmriprep')

    #space_label = 'T1w' #'MNI152NLin2009cAsym' # 'T1w' 

    task_label = task_id
    fwhm_sub = 1.5

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

    # check conditions
    print(models_events[0][3]['trial_type'].value_counts())
    #print(models_events[0][0]['feedback'].value_counts())

    # create stimulus list from updated events.tsv file
    stim_list = sorted([s for s in models_events[0][0]['trial_type'].unique() if str(s) != 'nan'])


    # #### Create trial-specific events

    for mx in range(len(models_events[0])):
        
        name_groups = models_events[0][mx].groupby('trial_type')['trial_type']
        suffix = name_groups.cumcount() + 1
        repeats = name_groups.transform('size')
        
        #models_events[0][mx]['trial_type'] = models_events[0][mx]['trial_type'] + '_trial' + \
        #                                     np.arange(len(models_events[0][mx])).astype(str)
        
        models_events[0][mx]['trial_type'] = models_events[0][mx]['trial_type'] + '_trial' + suffix.map(str)
        models_events[0][mx]['trial_type'] = models_events[0][mx]['trial_type'].str.replace('-','_')
        print(models_events[0][mx]['trial_type'])

    models_events[0][2]['trial_type']

    # create stimulus list from updated events.tsv file
    stim_list = sorted([s for s in models_events[0][mx]['trial_type'].unique() if str(s) != 'nan'])

    model_and_args = zip(models, models_run_imgs, models_events, models_confounds)

    # ### Across-runs GLM

    #contrast_label = 'sound' # 'right + wrong - none'
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

# ## plot tsnr
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


# #### Load subcortical (MNI space) regions
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


# ## Decoding

space_label = 'MNI152NLin2009cAsym' # 'T1w' 
nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                               'level-1_fwhm-%.02f'%fwhm_sub, 
                               'sub-%s_space-%s'%(model.subject_label, space_label))

z_maps = sorted(glob(nilearn_sub_dir+'/trial_models'+'/run*/*di*nii.gz'))
print('# of z-maps: ', len(z_maps))


#conditions = stim_list * len(imgs)
conditions = [os.path.basename(x)[68:74] for x in z_maps]
#conditions = [os.path.basename(x)[52:58] for x in z_maps]
print('# of conditions: ', len(conditions))

print(conditions)


conditions_tone = [stim[:3] for stim in conditions]

conditions_talker = [stim[4:6] for stim in conditions]

# ### Nilearn `Decoder` accuracies
# from https://nilearn.github.io/auto_examples/02_decoding/plot_haxby_glm_decoding.html#build-the-decoding-pipeline
# **note: does not generate predictions for each fold, so cannot use for confusion matrix generation**

from nilearn.decoding import Decoder

# available space-T1w masks: aud_mask_func_img, lh_mask_func_img, rh_mask_func_img
# available space-MNI masks: mask_IC_img
# mask = nib.load(os.path.join(nilearn_dir, 'sub-%s_mask-aud-ctx.nii.gz'%model.subject_label))
mask = mask_IC_fpath
decoder=Decoder(estimator='svc', mask=mask,
                standardize=False,
                screening_percentile=5, 
                cv=10,
               )
decoder.fit(z_maps, conditions)

print('# of CV folds: ', decoder.cv)
classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
chance_level = 1. / len(np.unique(conditions))
print('Classification accuracy: {:4f} / Chance level: {}'.format(
       classification_accuracy, chance_level))

from matplotlib import pyplot as plt
plt.figure(figsize=(6,3), dpi=150)
plt.boxplot(list(decoder.cv_scores_.values()));
plt.axhline(y=chance_level, color='r', linewidth=0.5)
plt.title('Inferior colliculus SVC accuracy: {:3f} (Chance level: {})'.format(
            classification_accuracy, chance_level))
#plt.xlabel('stimulus')
#plt.ylabel('accuracy')
plt.xticks(range(1,17), np.unique(conditions), rotation=30);
#plt.xticks(range(1,5), np.unique(conditions_tone));

print(decoder.coef_)

# #### Extract and save matrix

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

'''
# #### WIP TEST: Correlate stimulus maps to create (dis-)similarity matrix
from scipy.spatial import distance

func_files = glob(nilearn_sub_dir + '/*')
print(func_files)

dist_mat = np.zeros(len(func_files), len(func_files))
for fx, func_b_file in enumerate(func_files):
    pattern_a = nib.load(func_a_file).get_fdata()
    
    for gx, func_b_file in enumerate(func_files):
        pattern_b = nib.load(func_b_file).get_fdata()
        
        dst = distance.euclidean(pattern_a, pattern_b)
        dist_mat[fx,gx] = dst

# #### Confusion matrix

n_train = round(len(z_maps)*0.75)
n_test = len(z_maps) - n_train

# first, run a new classifier with a separate training and test set
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from nilearn.plotting import plot_matrix, show
from nilearn.image import index_img

cv = KFold(n_splits=5)
fold = 0
prediction = []
test_conds = []
for train, test in cv.split(conditions):
    fold += 1
    subdecode=Decoder(estimator='svc', mask=mask,
                      standardize=False,
                      screening_percentile=5, 
                      cv=0,
                      scoring='accuracy'
                   )
    # fit on the training set
    print('# of training images: ', n_train)
    print('# of testing images:  ', n_test)

    print('fitting the training data')
    subdecode.fit(index_img(z_maps, train), np.array(conditions)[train])
    
    # test
    print('predicting the test images')
    y_pred = subdecode.predict(index_img(z_maps, test))
    print(y_pred)
    prediction.append(y_pred)
    
    print(np.array(conditions)[test])
    test_conds.append(np.array(conditions)[test])

for fx in range(len(prediction)):
    print(prediction[fx]==test_conds[fx])

import seaborn as sns
plt.figure(figsize=(5,4),dpi=100)
confusion_mat = confusion_matrix(y_pred, conditions[n_train:], normalize='all')
ax = sns.heatmap(confusion_mat, 
                 cmap='hot_r',
                )

ax.set_xticks(range(len(np.unique(conditions))))
ax.set_yticks(range(len(np.unique(conditions))))
ax.set_xticklabels(np.unique(conditions), rotation=45)
ax.set_yticklabels(np.unique(conditions), rotation=45)

#ax.set_xticklabels(range(1,5))
#ax.set_yticklabels(range(1,5))

ax.set_xlabel('predicted')
ax.set_ylabel('true value')

plt.show()
print(confusion_mat)

# #### Plot decoder maps

t1w_mni_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/FLT/',
                             'derivatives', 'fmriprep',
                             'sub-%s/anat'%model.subject_label,
                             'sub-%s_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'%model.subject_label)

# plot
for stim in np.unique(conditions): #stim_list:
    decoder_img = decoder.coef_img_[stim]
    plotting.plot_stat_map(decoder_img, bg_img=t1w_mni_fpath, colorbar=True, 
                           title=stim,
                           cut_coords=[1, -35, -9]
                           #axes=axes[0],
                           #display_mode='x', cut_coords=6
                           )

plotting.show()
'''

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

'''
# ### Decoding

# In[ ]:


import nilearn.decoding
from nilearn.decoding import Decoder


# In[ ]:


models_events[0]


# In[ ]:


bold_list = [os.path.abspath(run) for run in models_run_imgs[0]]


# In[ ]:


print(models_events[0][0]['trial_type'])


# In[ ]:


runx = 0
fmri_img = nib.load(bold_list[runx])
conditions = models_events[0][runx]['trial_type']


# In[ ]:


conditions


# In[ ]:


mask_filename = striatum_mask_anat_img
decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True)


# In[ ]:


func_img=models_run_imgs[0][runx]


# In[ ]:


print(func_img)
print(nib.load(func_img).get_fdata().shape)


# In[ ]:


# 121 fmri timepoints, but only 120 trials. need to reduce
from nilearn.image import index_img
func_120_img = index_img(func_img, slice(0,120))


# In[ ]:


decoder.fit(func_120_img, conditions)


# In[ ]:


testx = 1
test_img = index_img(models_run_imgs[0][testx],slice(0,120))
test_conds = models_events[0][testx]['trial_type']


# In[ ]:


prediction = decoder.predict(test_img)
print(prediction)


# In[ ]:


print((prediction == test_conds).sum() / float(len(test_conds)))


# ## Searchlight decoding

# create mask
from nilearn.masking import compute_epi_mask
mask_img = compute_epi_mask(fmri_img)


from nilearn.image import resample_to_img

# create ROI mask
process_mask_img = lh_mask_img

# resample to functional space (was created from T1w freesurfer output)
process_mask_space_func_img = resample_to_img(process_mask_img, mask_img, interpolation='nearest')
process_mask = process_mask_space_func_img.get_fdata()


# In[ ]:


# create searchlight
searchlight = nilearn.decoding.SearchLight(mask_img, process_mask_space_func_img, radius=fwhm_sub, n_jobs=4, verbose=1)


# In[ ]:


# fit searchlight to image
searchlight.fit(fmri_img, y)


# In[ ]:


from nilearn.input_data import NiftiMasker
from nilearn.image import get_data
# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask_img=mask_img,
                           standardize=True, memory='nilearn_cache',
                           memory_level=1)
fmri_masked = nifti_masker.fit_transform(fmri_img)

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(fmri_masked, y)
p_values = -np.log10(p_values)
p_values[p_values > 10] = 10
p_unmasked = get_data(nifti_masker.inverse_transform(p_values))


# In[ ]:


min(p_values)


# In[ ]:


from nilearn import image
from nilearn.image import new_img_like
mean_fmri = image.mean_img(fmri_img)


from nilearn.plotting import plot_stat_map, plot_img, show
searchlight_img = new_img_like(mask_img, searchlight.scores_, affine=mask_img.affine)
#searchlight_resamp_img = resample_to_img(searchlight_img, mask_img)


# In[ ]:


print(searchlight.scores_.shape)
print(searchlight_img.shape)
print(mean_fmri.shape)
print(p_unmasked.shape)
print(process_mask.shape)


# In[ ]:


# Because scores are not a zero-center test statistics, we cannot use
# plot_stat_map
plot_img(searchlight_img, bg_img=mean_fmri,
         title="Searchlight", display_mode="z", cut_coords=[-10, -5, 0, 5, 10],
         vmin=.0, cmap='hot', threshold=1, black_bg=True)

# F_score results
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
f_score_img = new_img_like(mean_fmri, p_ma)
plot_stat_map(f_score_img, mean_fmri,
              title="F-scores", display_mode="z",
              cut_coords=[-10, -5, 0, 5, 10], alpha=.5,
              colorbar=True)

show()


# In[ ]:


# test mask images - should only be within STG
plotting.plot_roi(process_mask_space_func_img, bg_img=t1w_fpath, cmap='Paired')
plotting.plot_roi(process_mask_space_func_img, bg_img=mean_fmri, cmap='Paired')


# ### MNI space

space_label = 'MNI152NLin2009cAsym'

task_label = 'listen'
fwhm_sub = 3.75

models, models_run_imgs, models_events, models_confounds =     first_level_from_bids(data_dir, task_label, space_label,
                          smoothing_fwhm=fwhm_sub,
                          derivatives_folder=derivatives_folder)

# check model image files
print([os.path.basename(run) for run in models_run_imgs[0]])

# check model confounds
print(models_confounds[0][0].columns)

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

models_confounds[0][0][conf_keep_list]

# check conditions
print(models_events[0][0]['trial_type'].value_counts())

contrast_label = 'sound-silent'

fig, axes = plt.subplots(figsize=(12, 6))
model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
for midx, (model, imgs, events, confounds) in enumerate(model_and_args):    
    confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]
    # fit the GLM
    model.fit(imgs, events, confounds_ltd)
    # compute the contrast of interest
    zmap = model.compute_contrast(contrast_label)
    plotting.plot_glass_brain(zmap, colorbar=True, threshold=p001_unc,
                          title='sub-%s %s (unc p<%.03f; fwhm=%.02f)'%(model.subject_label, 
                                                                       task_label,p_val,fwhm_sub),
                          axes=axes,
                          plot_abs=False, display_mode='ortho')
    
    # save z map
    nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                   'level-1_fwhm-%.02f'%fwhm_sub, 'sub-%s'%model.subject_label)
    if not os.path.exists(nilearn_sub_dir):
        os.makedirs(nilearn_sub_dir)
    zmap_fpath = os.path.join(nilearn_sub_dir,
                              'sub-%s_task-%s_fwhm-%.02f_zmap.nii.gz'%(model.subject_label,task_label, fwhm_sub))
    nib.save(zmap, zmap_fpath)
    
    # save report
    report_fpath = os.path.join(nilearn_sub_dir,
                                'sub-%s_task-%s_fwhm-%.02f_report.html'%(model.subject_label,task_label, fwhm_sub))
    report = make_glm_report(model=model,
                             contrasts=contrast_label)
    report.save_as_html(report_fpath)
plotting.show()
# save plot
plot_fpath = os.path.join(nilearn_dir, 'sub-%s_task-%s_fwhm-%.02f.png'%(model.subject_label,task_label,fwhm_sub))
fig.savefig(plot_fpath)


# ### plot to surface

from nilearn.datasets import fetch_surf_fsaverage
fsaverage = fetch_surf_fsaverage(mesh='fsaverage5')

plotting.plot_img_on_surf(zmap, surf_mesh='fsaverage5', 
                          hemispheres=['left','right'], views=['lateral', 'medial'],
                          title="sub-%s task-%s fwhm-%.02fmm"%(model.subject_label, task_label, fwhm_sub),
                          threshold=2., colorbar=True,
                          output_filestr=os.path.join(nilearn_dir,
                                                      'sub-%s_task-%s_fwhm-%.02f_surface.png'%(model.subject_label, 
                                                                                               task_label,fwhm_sub)))


# ### plot model matrices
from nilearn.plotting import plot_contrast_matrix
plot_contrast_matrix(contrast_label, model.design_matrices_[0])

from nilearn.plotting import plot_design_matrix
plot_design_matrix(model.design_matrices_[0])
plt.show()
'''
