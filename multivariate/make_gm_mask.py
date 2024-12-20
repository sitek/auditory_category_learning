#!/bin/python

import os
import argparse
from glob import glob
import nibabel as nib

parser = argparse.ArgumentParser(
                description='Create subject-specific grey matter mask',
                epilog=('Example: python make_gm_mask.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

subject_id = args.sub
space_label=args.space
fwhm = args.fwhm
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

''' define other inputs '''
deriv_dir = os.path.join(bidsroot, 'derivatives')
nilearn_dir = os.path.join(deriv_dir, 'nilearn')

# masking function
def generate_mask(subject_id, fmriprep_dir, statmap_example_fpath, out_dir, space_label):
    from nilearn.image import resample_to_img
    
    # read in the overall brain mask
    anat_dir = os.path.join(fmriprep_dir,
                            'sub-{}/anat'.format(subject_id))

    # create binarized gray matter mask
    gm_fpath = os.path.join(anat_dir, f'sub-{subject_id}_space-{space_label}_label-GM_probseg.nii.gz')
    gm_img = nib.load(gm_fpath)
    
    from nilearn.image import binarize_img
    gm_bin_img = binarize_img(gm_img, threshold=0)
    mask_func_img = resample_to_img(gm_bin_img, statmap_example_fpath, interpolation='nearest')
    
    labelname = 'gm'
    out_fpath = os.path.join(out_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(subject_id, space_label, labelname))
    nib.save(mask_func_img, out_fpath)
    
    return out_fpath

''' run function '''
nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                           'level-1_fwhm-%.02f'%fwhm, 
                                           'sub-%s_space-%s'%(subject_id, space_label))    
statmap_example_fpath = sorted(glob(nilearn_sub_dir+'/*/run*/*di*.nii.gz'))[0]

sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%subject_id, 
                            'space-%s'%space_label, 'masks-dseg')   

if not os.path.exists(sub_mask_dir):
    os.makedirs(sub_mask_dir)

mask_fpath = generate_mask(subject_id, fmriprep_dir, statmap_example_fpath, sub_mask_dir, space_label)

