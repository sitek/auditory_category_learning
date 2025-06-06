
import os
import json
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd

from glob import glob

''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Extract ROI beta values',
                epilog=('Example: python mask_trial_betas.py --sub=FLT02 '
                        '--fwhm=0.00 --atlas=dseg --space=MNI152NLin2009cAsym '
                        '--stat=tstat --model=run-all_LSS --task=tonecat')
                )

parser.add_argument("--sub", 
                    help="participant id", type=str)
parser.add_argument("--atlas",
                    help="pre-extracted atlas", type=str)
parser.add_argument("--space", 
                    help="space label", type=str)
parser.add_argument("--fwhm", 
                    help="spatial smoothing full-width half-max", 
                    type=float)
parser.add_argument("--stat", 
                    help="stat file (options: beta, tstat, zstat)", 
                    type=str)
parser.add_argument("--model", 
                    help=("which model to operate on "
                          "(options: run-all, stimulus_per_run, trial_models)"), 
                    type=str)
parser.add_argument("--task", 
                    help="task id", type=str)
parser.add_argument("--mask_dir", 
                    help="directory containing subdirectories with masks for each subject", 
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
    
subject_id  = args.sub
atlas_label = args.atlas
fwhm_sub    = args.fwhm
space_label = args.space
stat_descrip = args.stat
model_descrip = args.model
task_label = args.task
mask_dir     = args.mask_dir
bidsroot     = args.bidsroot
fmriprep_dir = args.fmriprep_dir

nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')
print(nilearn_dir)

roi_dict_MNI_dseg = {'L-Caud': 35, 'L-Put': 36, #'L-Pall': 37, 'L-Accumb': 41, 
                     'L-HG': 189, 'L-PP': 187, 'L-PT': 191, 
                     'L-STGa': 117, 'L-STGp': 119, 'L-ParsOp': 111, 'L-ParsTri': 109, 
                     #'L-SMGa': 137, 'L-SMGp': 139, 'L-Ang': 141, 
                     'R-Caud': 46, 'R-Put': 47, #'R-Pall': 48, 'R-Accumb': 45, 
                     'R-HG': 190, 'R-PP': 188, 'R-PT': 192, 
                     'R-STGa': 118, 'R-STGp': 120, 'R-ParsOp': 112, 'R-ParsTri': 110,
                     #'R-SMGa': 138, 'R-SMGp': 140, 'R-Ang': 142, 
                    }
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


def mask_fmri(fmri_niimgs, mask_filename, fwhm):
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img=mask_filename, #runs=session_label,
                         smoothing_fwhm=fwhm, standardize=True,
                         #memory="nilearn_cache", memory_level=1
                        )
    fmri_masked = masker.fit_transform(fmri_niimgs)
    return fmri_masked, masker

if atlas_label == 'subcort-aud':
    roi_list = list(roi_dict_MNI_sg_subcort.keys()) 
elif atlas_label == 'dseg':
    roi_list = list(roi_dict_MNI_dseg.keys())
elif atlas_label == 'tian-S3':
    roi_list = list(roi_dict_tian_S3.keys())
print(roi_list)

nilearn_sub_dir = os.path.join(nilearn_dir, 
                               'bids-deriv_level-1_fwhm-%.02f'%fwhm_sub, 
                               f'sub-{subject_id}_space-{space_label}', 
                               model_descrip, 
                              )
print(nilearn_sub_dir)

run_dir = os.path.join(nilearn_sub_dir, 'run-grouped')

for mx, mask_descrip in enumerate(roi_list):
    print(mask_descrip)
    # define the mask for the region of interest
    masks_dir = os.path.join(mask_dir, 
                             f'sub-{subject_id}', 
                             f'space-{space_label}', 
                             f'masks-{atlas_label}')
    mask_fpath = os.path.join(masks_dir, 
                              f'sub-{subject_id}_space-{space_label}_mask-{mask_descrip}.nii.gz')

    # run-grouped
    for rx, rungroup in enumerate(['early', 'middle', 'final']):
        print(rungroup)
        stat_maps = sorted(glob(run_dir+f'/*rungroup-{rungroup}*Di*stat-{stat_descrip}_statmap.nii.gz')) 
        print('# of stat maps: ', len(stat_maps))    

        conditions_all = [(os.path.basename(x).split('_')[2]) for x in (stat_maps)]
        print('conditions_all:', conditions_all)


        out_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                               f'bids-deriv_level-1_fwhm-{fwhm_sub:.02f}',
                               'masked_statmaps',
                               f'sub-{subject_id}', 
                               'statmaps_masked',
                               model_descrip,
                               f'rungroup-{rungroup}',
                               f'mask-{mask_descrip}')
        os.makedirs(out_dir, exist_ok=True)

        for sx, stat_fpath in enumerate(stat_maps):
            cond_label = conditions_all[sx]
            print('cond_label:', cond_label)
            fmri_masked, masker = mask_fmri(stat_fpath, mask_fpath, fwhm_sub)

            # save out
            out_fbase = f'sub-{subject_id}_space-{space_label}_mask-{mask_descrip}_rungroup-{rungroup}_cond-{cond_label}_map-{stat_descrip}'

            out_fpath = os.path.join(out_dir, out_fbase+'.csv')
            np.savetxt(out_fpath, fmri_masked)

            masked_out_fpath = os.path.join(out_dir, out_fbase+'.nii.gz')
            masked_img = masker.inverse_transform(fmri_masked)
            nib.save(masked_img, masked_out_fpath)