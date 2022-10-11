
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib

from glob import glob
from nilearn.image import new_img_like


parser = argparse.ArgumentParser(
                description='Subject-level multivariate searchlight analysis',
                epilog='Example: python multivariate_searchlight.py --sub=FLT02 --space=MNI152NLin2009cAsym --fwhm=1.5 --cond=tone'
        )

parser.add_argument("--sub", help="participant id", type=str)
parser.add_argument("--space", help="space label", type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", type=float)
parser.add_argument("--cond", help="condition to analyze", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
subject_id = args.sub
space_label=args.space
fwhm = args.fwhm
cond_label = args.cond

''' define other inputs '''
project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
fmriprep_dir = os.path.join(project_dir, 'derivatives', 'fmriprep_noSDC')

bidsroot = os.path.join(project_dir, 'data_bids_noIntendedFor')
deriv_dir = os.path.join(bidsroot, 'derivatives')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
print(nilearn_dir)

roi_dict_MNI_dseg = {'L-Caud': 35, 'L-Put': 36, 'L-HG': 189, 'L-PP': 187, 'L-PT': 191, 
                     'L-STGa': 117, 'L-STGp': 119, 'L-ParsOp': 111, 'L-ParsTri': 109, 
                     'R-Caud': 46, 'R-Put': 47, 'R-HG': 190, 'R-PP': 188, 'R-PT': 192, 
                     'R-STGa': 118, 'R-STGp': 120, 'R-ParsOp': 112, 'R-ParsTri': 110, }
roi_dict_T1w_aseg = {'L-VentralDC': 28, 'L-Caud': 11, 'L-Put': 12, 'L-HG': 1034, 'L-STG': 1030, 
                     'L-ParsOp': 1018, 'L-ParsTri': 1020, 'L-SFG': 1028, 'Brainstem': 16, 
                     'R-VentralDC': 60, 'R-Caud': 50, 'R-Put': 51, 'R-HG': 2034, 'R-STG': 2030, 
                     'R-ParsOp': 2018, 'R-ParsTri': 2020, 'R-SFG': 2028, 'CSF': 24}

roi_dict = roi_dict_MNI_dseg # roi_dict_T1w_aseg # 

def create_labels(stat_maps):
    import os
    from glob import glob
    from numpy.random import shuffle
    from copy import copy
    
    # 16 stimulus decoding
    conditions_all = ['_'.join(os.path.basename(x).split('_')[5:8]) for x in (stat_maps)]
    #print('all events: ', conditions_all[:10])

    # 4-category decoding
    conditions_tone = [os.path.basename(x).split('_')[5] for x in (stat_maps)]
    print('tone conditions: ', np.unique(conditions_tone))

    conditions_talker = [os.path.basename(x).split('_')[6] for x in (stat_maps)]
    print('talker conditions: ', np.unique(conditions_talker))
    
    # shuffled conditions
    conditions_shuffled = copy(conditions_tone)
    shuffle(conditions_shuffled)
    print('tone conditions: ', np.unique(conditions_shuffled))

    return conditions_tone, conditions_talker, conditions_all, conditions_shuffled

def fit_searchlight(region_mask, brain_mask, fmri_img, y):
    # Define the cross-validation scheme used for validation.
    # Here we use a KFold cross-validation on the session, which corresponds to
    # splitting the samples in 4 folds and make 4 runs using each fold as a test
    # set once and the others as learning sets
    from sklearn.model_selection import KFold, StratifiedKFold
    cv = StratifiedKFold(n_splits=4)

    import nilearn.decoding
    # The radius is the one of the Searchlight sphere that will scan the volume
    searchlight = nilearn.decoding.SearchLight(
        brain_mask,
        process_mask_img=region_mask,
        radius=4.5,
        verbose=1, cv=cv, n_jobs=-1)
    searchlight.fit(fmri_img, y)
    
    return searchlight

''' run the pipeline '''
roi_list = list(roi_dict.keys())
print(roi_list)
num_rois = len(roi_list)

nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                               'level-1_fwhm-%.02f'%fwhm, 
                               'sub-%s_space-%s'%(subject_id, space_label))
print(nilearn_sub_dir)

# run-specific stimulus beta maps
stat_maps = sorted(glob(nilearn_sub_dir+'/trial_models'+'/run*/*di*nii.gz')) 
print('# of stat maps: ', len(stat_maps))   

f_affine = nib.load(stat_maps[0]).affine

# read in the overall brain mask
anat_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/FLT/',
                        'derivatives/fmriprep_noSDC/sub-{}/anat'.format(subject_id))
brainmask_fpath = os.path.join(anat_dir, 'sub-{}_space-{}_desc-brain_mask.nii.gz'.format(subject_id, space_label))

# generate condition labels based on filenames
conditions_tone, conditions_talker, conditions_all, conditions_shuffled = create_labels(stat_maps)

# create output directory
sub_out_dir = os.path.join(nilearn_sub_dir, 'searchlight')
if not os.path.exists(sub_out_dir):
    os.makedirs(sub_out_dir)

#for mx, mask_descrip in enumerate(roi_list):
# define the mask for the region of interest

# use the whole brain mask
mask_descrip = 'gm-thr90'
masks_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%subject_id, 'space-%s'%space_label, 'masks-dseg', )
mask_fpath = os.path.join(masks_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(subject_id, space_label, mask_descrip))

# run the searchlight
print('running searchlight on {} with mask {} and labels {}'.format(subject_id, mask_descrip, cond_label))
if cond_label == 'tone':
    searchlight = fit_searchlight(mask_fpath, brainmask_fpath, stat_maps, conditions_tone)
elif cond_label == 'shuffled':
    searchlight = fit_searchlight(mask_fpath, brainmask_fpath, stat_maps, conditions_shuffled)

# turn searchlight scores into a 3D brain image
searchlight_img = new_img_like(stat_maps[0], searchlight.scores_, ) # affine=f_affine)

# save to an output file
out_fpath = os.path.join(sub_out_dir, 
                         'sub-{}_space-{}_mask-{}_cond-{}_searchlight.nii.gz'.format(subject_id, space_label, 
                                                                                     mask_descrip, cond_label))
nib.save(searchlight_img, out_fpath)
print('saved searchlight image to ', out_fpath)

