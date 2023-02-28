import os
import json
import numpy as np
import nibabel as nib

from glob import glob

project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
fmriprep_dir = os.path.join(project_dir, 'derivatives', 'fmriprep_noSDC')

bidsroot = os.path.join(project_dir, 'data_bids_noIntendedFor')
deriv_dir = os.path.join(bidsroot, 'derivatives')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
print(nilearn_dir)

task_list = ['tonecat']

sub_list_mand = ['FLT01', 'FLT03', 'FLT05', 'FLT07', 'FLT08', 'FLT10', ]

sub_list_nman = ['FLT02', 'FLT04', 'FLT06', 'FLT09', 'FLT11', 'FLT12', 'FLT13', ]

sub_list = sub_list_mand + sub_list_nman

roi_dict_T1w_aseg = {'L-VentralDC': 28, 'L-Caud': 11, 'L-Put': 12, 'L-HG': 1034, 'L-STG': 1030, 'L-ParsOp': 1018, 'L-ParsTri': 1020, 'L-SFG': 1028,
                     'R-VentralDC': 60, 'R-Caud': 50, 'R-Put': 51, 'R-HG': 2034, 'R-STG': 2030, 'R-ParsOp': 2018, 'R-ParsTri': 2020, 'R-SFG': 2028,}

roi_dict = roi_dict_T1w_aseg

def create_labels(z_maps):
    import os
    from glob import glob
    from numpy.random import shuffle
    from copy import copy
    
    # 16 stimulus decoding
    #conditions_all = [os.path.basename(x)[-31:-18] for x in z_maps] 
    conditions_all = ['_'.join(os.path.basename(x).split('_')[5:9]) for x in (z_maps)]

    # 4-category decoding
    #conditions_tone = [stim[:3] for stim in conditions_all]
    conditions_tone = [os.path.basename(x).split('_')[5] for x in (z_maps)]
    print('tone conditions: ', np.unique(conditions_tone))

    #conditions_talker = [stim[4:6] for stim in conditions_all]
    conditions_talker = [os.path.basename(x).split('_')[6] for x in (z_maps)]
    print('talker conditions: ', np.unique(conditions_talker))

    
    # shuffled conditions
    conditions_shuffled = copy(conditions_tone)
    shuffle(conditions_shuffled)
    print('tone conditions: ', np.unique(conditions_shuffled))

    return conditions_tone, conditions_talker, conditions_shuffled

def mask_fmri(fmri_niimgs, mask_filename, fwhm):
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img=mask_filename, #runs=session_label,
                         smoothing_fwhm=fwhm, standardize=True,
                         memory="nilearn_cache", memory_level=1)
    fmri_masked = masker.fit_transform(fmri_niimgs)
    return fmri_masked, masker

''' Subject- and region-specific confusion matrices '''
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

fwhm_sub = 1.5
space_label = 'T1w'

roi_list = list(roi_dict.keys())
print(roi_list)

mand_conf_mat = []
nman_conf_mat = []
for sub_id in sub_list:
    print(sub_id)
    sub_conf_mat = []
    
    fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(16,6), dpi=300)
    fig.suptitle(sub_id)
    
    for mx, mask_descrip in enumerate(roi_list):
        # define the mask for the region of interest
        print(mask_descrip)
        masks_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 'space-%s'%space_label, 'masks-dseg', )
        mask_fpath = os.path.join(masks_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(sub_id, space_label, mask_descrip))
        #mask_img = nib.load(mask_fpath)

        nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                       'level-1_fwhm-%.02f'%fwhm_sub, 
                                       'sub-%s_space-%s'%(sub_id, space_label))

        # run-specific stimulus beta maps
        stat_maps = sorted(glob(nilearn_sub_dir+'/stimulus_per_run'+'/run*/*di*nii.gz')) 
        print('# of stat maps: ', len(stat_maps))    
        
        fmri_masked, masker = mask_fmri(stat_maps, mask_fpath, fwhm_sub)

        conditions_tone, conditions_talker, conditions_shuffled = create_labels(stat_maps)
        
        # Split the data into a training set and a test set
        x, y = fmri_masked, conditions_tone
        #x, y = fmri_masked, conditions_shuffled
        
        X_train, X_test, y_train, y_test = train_test_split(x,y, 
                                                            test_size=0.25,
                                                            random_state=0, 
                                                            stratify=y)

        # Run classifier, using a model that is too regularized (C too low) to see
        # the impact on the results
        clf = svm.SVC(kernel="linear", ).fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        
        ax = plt.subplot(2, 8, mx + 1)
        cm_display = ConfusionMatrixDisplay(cm,).plot(ax=ax, colorbar=False, 
                                                      im_kw={'cmap':'Blues'})
        ax.set_title(mask_descrip)
        
        sub_conf_mat.append(cm)
    # better spacing
    fig.tight_layout()
    fig.savefig('sub-{}_confusion_matrices.png'.format(sub_id))
    #fig.savefig('sub-{}_shuffled_confusion_matrices.png'.format(sub_id))

    # append to group list
    if sub_id in sub_list_mand:
        mand_conf_mat.append(sub_conf_mat)
    elif sub_id in sub_list_nman:
        nman_conf_mat.append(sub_conf_mat)

        
''' Group mean plots '''
mand_mean_cm = np.mean(mand_conf_mat, axis=0)
nman_mean_cm = np.mean(nman_conf_mat, axis=0)

np.save('mand_mean_cm', mand_mean_cm)
np.save('nman_mean_cm', nman_mean_cm)

fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(16,6), dpi=300)
fig.suptitle('Mandarin-speaking mean confusion matrices')

for mx, mask_descrip in enumerate(roi_list):
    ax = plt.subplot(2, 8, mx + 1)
    ConfusionMatrixDisplay(mand_mean_cm[mx],).plot(ax=ax, colorbar=False, 
                                                   im_kw={'cmap':'Blues'})
    ax.set_title(mask_descrip)
# better spacing
fig.tight_layout()
fig.savefig('group-Mand_confusion_matrices.png')
#fig.savefig('group-Mand_shuffled_confusion_matrices.png')
    
fig2, axs = plt.subplots(nrows=2, ncols=8, figsize=(16,6), dpi=300)
fig2.suptitle('Non-Mandarin-speaking mean confusion matrices')

for mx, mask_descrip in enumerate(roi_list):
    ax = plt.subplot(2, 8, mx + 1)
    ConfusionMatrixDisplay(nman_mean_cm[mx],).plot(ax=ax, colorbar=False, 
                                                   im_kw={'cmap':'Blues'})
    ax.set_title(mask_descrip)
# better spacing
fig2.tight_layout()
fig2.savefig('group-NMan_confusion_matrices.png')
#fig2.savefig('group-NMan_shuffled_confusion_matrices.png')