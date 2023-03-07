
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
                epilog=('Example: python multivariate_searchlight.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 --cond=tone --searchrad=4.5'
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
        )

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--cond", help="condition to analyze", 
                    type=str)
parser.add_argument("--searchrad", help="searchlight radius (in voxels)", 
                    type=str)
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
space_label=args.space
fwhm = args.fwhm
cond_label = args.cond
searchlight_radius = args.searchrad
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

''' define other inputs '''
nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')

def create_labels(stat_maps):
    import os
    from glob import glob
    from numpy.random import shuffle
    from copy import copy
    
    # all-stimulus decoding
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
    print('shuffled conditions: ', np.unique(conditions_shuffled))

    return conditions_tone, conditions_talker, conditions_all, conditions_shuffled

def fit_searchlight(region_mask, brain_mask, fmri_img, y, searchlight_radius=searchlight_radius):
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
        radius=searchlight_radius,
        verbose=1, cv=cv, n_jobs=-1)
    searchlight.fit(fmri_img, y)
    
    return searchlight

''' run the pipeline '''
nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                               'level-1_fwhm-%.02f'%fwhm, 
                               'sub-%s_space-%s'%(subject_id, space_label))

# run-specific stimulus beta maps
stat_maps = sorted(glob(nilearn_sub_dir+'/stimulus_per_run'+'/run*/*di*map-beta.nii.gz')) 
print('# of stat maps: ', len(stat_maps))   

f_affine = nib.load(stat_maps[0]).affine

# read in the overall brain mask
anat_dir = os.path.join(fmriprep_dir, 'sub-{}/anat'.format(subject_id))
brainmask_fpath = os.path.join(anat_dir, 'sub-{}_space-{}_desc-brain_mask.nii.gz'.format(subject_id, space_label))

# generate condition labels based on filenames
conditions_tone, conditions_talker, conditions_all, conditions_shuffled = create_labels(stat_maps)

# create output directory
sub_out_dir = os.path.join(nilearn_sub_dir, 'searchlight_radius-{}'.format(searchlight_radius))
if not os.path.exists(sub_out_dir):
    os.makedirs(sub_out_dir)

# use the whole brain gray matter mask
#mask_descrip = 'gm-thr90'
mask_descrip = 'gm'
masks_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%subject_id, 'space-%s'%space_label, 'masks-dseg', )
mask_fpath = os.path.join(masks_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(subject_id, space_label, mask_descrip))

# run the searchlight
print('running searchlight on {} with mask {} and labels {}'.format(subject_id, mask_descrip, cond_label))
if cond_label == 'tone':
    searchlight = fit_searchlight(mask_fpath, brainmask_fpath, stat_maps, 
                                  conditions_tone, searchlight_radius)
elif cond_label == 'shuffled':
    searchlight = fit_searchlight(mask_fpath, brainmask_fpath, stat_maps, 
                                  conditions_shuffled, searchlight_radius)

# turn searchlight scores into a 3D brain image
searchlight_img = new_img_like(stat_maps[0], searchlight.scores_, ) # affine=f_affine)

# save to an output file
out_fpath = os.path.join(sub_out_dir, 
                         'sub-{}_space-{}_mask-{}_cond-{}_searchlight.nii.gz'.format(subject_id, space_label, 
                                                                                     mask_descrip, cond_label))
nib.save(searchlight_img, out_fpath)
print('saved searchlight image to ', out_fpath)

