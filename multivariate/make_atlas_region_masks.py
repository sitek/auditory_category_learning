#!/bin/python

import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
from glob import glob

parser = argparse.ArgumentParser(
                description='Create subject-specific grey matter mask',
                epilog=('Example: python make_gm_mask.py --sub=FLT02 '
                        ' --space=MNI152NLin2009cAsym --fwhm=1.5 '
                        ' --atlas_label=subcort_aud '
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--space", help="space label", 
                    type=str)
parser.add_argument("--fwhm", help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--atlas_label", 
                    help=("name of custom atlas label (options: "
                          " `subcort_aud`, `carpet_dseg`, `aparc`, `tian_s3', `carpet_motor`"), 
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
    
sub_id = args.sub
space_label=args.space
fwhm = args.fwhm
atlas_label = args.atlas_label
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

''' define other inputs '''
deriv_dir = os.path.join(bidsroot, 'derivatives')
nilearn_dir = os.path.join(deriv_dir, 'nilearn')

''' define atlas region dictionaries '''
roi_dict_MNI_dseg = {'L-Caud': 35, 'L-Put': 36, 'L-Pall': 37, 'L-Accumb': 41, 
                     'L-HG': 189, 'L-PP': 187, 
                     'L-PT': 191, 'L-STGa': 117, 'L-STGp': 119, 
                     'L-ParsOp': 111, 'L-ParsTri': 109, 
                     'L-SMGa': 137, 'L-SMGp': 139, 'L-Ang': 141,
                     'L-Lat-Ventricle': 3, 'L-Supracalcarine': 193,
                     'R-Caud': 46, 'R-Put': 47, 'R-Pall': 48, 'R-Accumb': 45, 
                     'R-HG': 190, 'R-PP': 188, 
                     'R-PT': 192, 'R-STGa': 118, 'R-STGp': 120, 
                     'R-ParsOp': 112, 'R-ParsTri': 110, 
                     'R-Lat-Ventricle': 4, 'R-Supracalcarine': 194,
                     'R-SMGa': 138, 'R-SMGp': 140, 'R-Ang': 142, 
                    }
roi_dict_MNI_motor = {'L-Caud': 35, 'L-Put': 36, 
                      'R-Caud': 46, 'R-Put': 47, 
                      'L-Precentral': 113, 'R-Precentral': 114,
                      'L-Postcentral': 133, 'R-Postcentral': 134,
                      'L-SMA': 151, 'R-SMA': 152,
                      'CerebMidb': 255,
                      }
roi_dict_T1w_aseg = {'L-VentralDC': 28, 'L-Caud': 11, 'L-Put': 12, 
                     'L-HG': 1034, 'L-STG': 1030, 'L-ParsOp': 1018, 
                     'L-ParsTri': 1020, 'L-SFG': 1028, 'Brainstem': 16, 
                     'R-VentralDC': 60, 'R-Caud': 50, 'R-Put': 51, 
                     'R-HG': 2034, 'R-STG': 2030, 'R-ParsOp': 2018, 
                     'R-ParsTri': 2020, 'R-SFG': 2028, 'CSF': 24}
roi_dict_MNI_sg_subcort = {'L-CN': 1, 'L-SOC': 3, 'L-IC': 5, 'L-MGN': 7, 
                           'R-CN': 2, 'R-SOC': 4, 'R-IC': 6, 'R-MGN': 8, }

roi_dict_tian_S3 = {}
tian_sc_S3_roi_list = ['HIP-head-m-rh','HIP-head-l-rh','HIP-body-m-rh','HIP-body-l-rh',
                        'AMY-SM-rh','AMY-CL-rh',
                        'THA-DPl-rh','THA-DPm-rh',
                        'THA-VPm-rh','THA-VPl-rh',
                        'THA-VAi-rh','THA-VAs-rh',
                        'THA-DAm-rh','THA-DAl-rh',
                        'PUT-VA-rh','PUT-DA-rh',
                        'PUT-VP-rh','PUT-DP-rh',
                        'CAU-VA-rh','CAU-DA-rh',
                        'HIP-tail-rh','lAMY-rh',
                        'pGP-rh','aGP-rh',
                        'NAc-shell-rh','NAc-core-rh','pCAU-rh',
                        'HIP-head-m-lh','HIP-head-l-lh','HIP-body-m-lh','HIP-body-l-lh',
                        'AMY-SM-lh','AMY-CL-lh',
                        'THA-DPl-lh','THA-DPm-lh',
                        'THA-VPm-lh','THA-VPl-lh',
                        'THA-VAi-lh','THA-VAs-lh',
                        'THA-DAm-lh','THA-DAl-lh',
                        'PUT-VA-lh','PUT-DA-lh',
                        'PUT-VP-lh', 'PUT-DP-lh',
                        'CAU-VA-lh','CAU-DA-lh',
                        'HIP-tail-lh','lAMY-lh',
                        'pGP-lh','aGP-lh',
                        'NAc-shell-lh','NAc-core-lh','pCAU-lh'
                        ]
for rx, roi in enumerate(tian_sc_S3_roi_list):
    roi_dict_tian_S3[roi] = rx+1

''' mask function '''
def generate_mask(sub_id, zmap_example_fpath, atlas_fpath, labelnum, labelname, out_dir, spacelabel):
    from nilearn.image import resample_to_img
    
    atlas_img = nib.load(atlas_fpath)
    atlas_data = atlas_img.get_fdata()
    atlas_affine = atlas_img.affine
    
    mask_data = np.zeros((atlas_data.shape))
    mask_data[np.where(atlas_data == labelnum)] = 1

    mask_img = nib.Nifti1Image(mask_data, atlas_affine)

    mask_func_img = resample_to_img(mask_img, zmap_example_fpath, interpolation='nearest')
    
    out_fpath = os.path.join(out_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(sub_id, space_label, labelname))
    nib.save(mask_func_img, out_fpath)
    
    return out_fpath

''' create atlas region masks '''
nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                           'level-1_fwhm-%.02f'%fwhm, 
                                           'sub-%s_space-%s'%(sub_id, space_label))    
print(nilearn_sub_dir)
z_maps = sorted(glob(nilearn_sub_dir+'/stimulus_per_run*/run*/*di*.nii.gz'))
zmap_example_fpath = z_maps[0]

if space_label == 'T1w' and atlas_label == 'aparc': 
    atlas_fpath = os.path.join(fmriprep_dir, 'sub-%s'%sub_id, 'anat',
                                'sub-%s_desc-%saseg_dseg.nii.gz'%(sub_id, atlas_label))


    sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 
                                'space-%s'%space_label, 'masks-aparc') 
    roi_dict = roi_dict_T1w_aseg
elif space_label == 'MNI152NLin2009cAsym' and atlas_label == 'carpet_dseg': 
    atlas_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/',
                               'reference/', #tpl-MNI152NLin2009cAsym/',
                               'tpl-MNI152NLin2009cAsym_res-01_desc-carpet_dseg.nii.gz')  
    sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 
                                'space-%s'%space_label, 'masks-dseg')  
    roi_dict = roi_dict_MNI_dseg
elif space_label == 'MNI152NLin2009cAsym' and atlas_label == 'carpet_motor': 
    atlas_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/',
                               'reference/', 
                               'tpl-MNI152NLin2009cAsym_res-01_desc-carpet_dseg.nii.gz')  
    sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 
                                'space-%s'%space_label, 'masks-dseg-motor')  
    roi_dict = roi_dict_MNI_motor
elif space_label == 'MNI152NLin2009cAsym' and atlas_label == 'subcort_aud':
    atlas_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/',
                               'reference/MNI_space/atlases',
                               'sub-bigbrain_MNI_conjunction_rois.nii.gz')
    sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 
                                'space-%s'%space_label, 'masks-subcort-aud') 
    roi_dict = roi_dict_MNI_sg_subcort
elif space_label == 'MNI152NLin2009cAsym' and atlas_label == 'tian_S3':
    atlas_fpath = os.path.join('/bgfs/bchandrasekaran/krs228/data/',
                               'reference/subcortex/Group-Parcellation/7T',
                               'Tian_Subcortex_S3_7T.nii')
    sub_mask_dir = os.path.join(nilearn_dir, 'masks', 'sub-%s'%sub_id, 
                                'space-%s'%space_label, 'masks-tian-S3') 
    roi_dict = roi_dict_tian_S3
else:
    print('mismatch between space label and atlas label')
if not os.path.exists(sub_mask_dir):
    os.makedirs(sub_mask_dir)

for key, value in roi_dict.items():
    print('generating {} mask file'.format(key))
    mask_fpath = generate_mask(sub_id, zmap_example_fpath, atlas_fpath, 
                               value, key, sub_mask_dir, space_label)
