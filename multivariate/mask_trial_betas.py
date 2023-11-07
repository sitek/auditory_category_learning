
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
'''
parser.add_argument("--event_type", 
                    help="what to model (options: `sound` or `stimulus` or `feedback`)", 
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
'''
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

project_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/', 'FLT/')
fmriprep_dir = os.path.join(project_dir, 'derivatives', '22.1.1')

bidsroot = os.path.join(project_dir, 'data_bids')
deriv_dir = os.path.join(bidsroot, 'derivatives')

nilearn_dir = os.path.join(deriv_dir, 'nilearn')
print(nilearn_dir)

task_list = ['tonecat']


roi_dict_MNI_dseg = {'L-Caud': 35, 'L-Put': 36, 'L-HG': 189, 'L-PP': 187, 'L-PT': 191, 
                     'L-STGa': 117, 'L-STGp': 119, 'L-ParsOp': 111, 'L-ParsTri': 109, 
                     'R-Caud': 46, 'R-Put': 47, 'R-HG': 190, 'R-PP': 188, 'R-PT': 192, 
                     'R-STGa': 118, 'R-STGp': 120, 'R-ParsOp': 112, 'R-ParsTri': 110, }
roi_dict_MNI_sg_subcort = {'L-CN': 1, 'L-SOC': 3, 'L-IC': 5, 'L-MGN': 7, 
                           'R-CN': 2, 'R-SOC': 4, 'R-IC': 6, 'R-MGN': 8, }


def mask_fmri(fmri_niimgs, mask_filename, fwhm):
    from nilearn.maskers import NiftiMasker
    masker = NiftiMasker(mask_img=mask_filename, #runs=session_label,
                         smoothing_fwhm=fwhm, standardize=True,
                         memory="nilearn_cache", memory_level=1)
    fmri_masked = masker.fit_transform(fmri_niimgs)
    return fmri_masked, masker

if atlas_label == 'subcort-aud':
    roi_list = list(roi_dict_MNI_sg_subcort.keys()) 
elif atlas_label == 'dseg':
    roi_list = ['L-HG', 'L-PP', 'L-PT', 'L-STGa', 'L-STGp', 'L-ParsOp', 'L-ParsTri', 
                'R-HG', 'R-PP', 'R-PT', 'R-STGa', 'R-STGp', 'R-ParsOp', 'R-ParsTri', ]

print(roi_list)

nilearn_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                               'level-1_fwhm-%.02f'%fwhm_sub, 
                               'sub-%s_space-%s'%(subject_id, space_label), 
                               model_descrip, 
                              )
print(nilearn_sub_dir)

for mx, mask_descrip in enumerate(roi_list):
    print(mask_descrip)
    # define the mask for the region of interest
    masks_dir = os.path.join(nilearn_dir, 'masks', 'sub-{}'.format(subject_id), 
                             'space-{}'.format(space_label), 
                             'masks-{}'.format(atlas_label))
    mask_fpath = os.path.join(masks_dir, 'sub-%s_space-%s_mask-%s.nii.gz'%(subject_id, space_label, mask_descrip))

    # run-all stat maps
    if 'run-all' in model_descrip:
        stat_maps = sorted(glob(nilearn_sub_dir+'/*di*map-{}.nii.gz'.format(stat_descrip))) 
        print('# of stat maps: ', len(stat_maps))    

        conditions_all = ['_'.join(os.path.basename(x).split('_')[5:8]) for x in (stat_maps)]


        out_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                               'level-1_fwhm-%.02f'%fwhm_sub,
                               'masked_statmaps',
                               'sub-{}'.format(subject_id), 
                               'statmaps_masked',
                               model_descrip,
                               'mask-{}'.format(mask_descrip ))
        os.makedirs(out_dir, exist_ok=True)

        for sx, stat_fpath in enumerate(stat_maps):
            cond_label = conditions_all[sx]
            #print(cond_label)

            #print('masking ', mask_descrip)
            fmri_masked, masker = mask_fmri(stat_fpath, mask_fpath, fwhm_sub)

            # save out
            out_fname = 'sub-%s_space-%s_mask-%s_run-all_cond-%s_map-%s.csv'%(subject_id, space_label, mask_descrip, 
                                                                              cond_label, stat_descrip)
            out_fpath = os.path.join(out_dir, out_fname)
            np.savetxt(out_fpath, fmri_masked)
            #print('ROI-masked {}-stats saved to '.format(stat_descrip), out_fpath)

            '''
            masked_out_fpath = os.path.join(out_dir, 'sub-%s_space-%s_mask-%s_run-all_cond-%s_map-%s.nii.gz'%(subject_id, space_label, mask_descrip, 
                                                                                                              cond_label, stat_descrip))
            masked_img = masker.inverse_transform(fmri_masked)
            nib.save(masked_img, masked_out_fpath)
            '''

    # run-specific stimulus stat maps
    elif 'stimulus_per_run' in model_descrip:
        for rx, run_dir in enumerate(sorted(glob(nilearn_sub_dir+'/run*'))):
            stat_maps = sorted(glob(run_dir+'/*di*map-{}.nii.gz'.format(stat_descrip))) 
            print('# of stat maps: ', len(stat_maps))    

            conditions_all = ['_'.join(os.path.basename(x).split('_')[5:8]) for x in (stat_maps)]


            out_dir = os.path.join(bidsroot, 'derivatives', 'nilearn',
                                   'level-1_fwhm-%.02f'%fwhm_sub,
                                   'masked_statmaps',
                                   'sub-{}'.format(subject_id), 
                                   'statmaps_masked',
                                   model_descrip,
                                   'run%02d'%rx,
                                   'mask-{}'.format(mask_descrip ))
            os.makedirs(out_dir, exist_ok=True)

            for sx, stat_fpath in enumerate(stat_maps):
                cond_label = conditions_all[sx]
                #print(cond_label)

                #print('masking ', mask_descrip)
                fmri_masked, masker = mask_fmri(stat_fpath, mask_fpath, fwhm_sub)

                # save out
                out_fname = 'sub-%s_space-%s_mask-%s_run-%02d_cond-%s_map-%s.csv'%(subject_id, space_label, mask_descrip, 
                                                                                   rx, cond_label, stat_descrip)
                out_fpath = os.path.join(out_dir, out_fname)
                np.savetxt(out_fpath, fmri_masked)
                #print('ROI-masked {}-stats saved to '.format(stat_descrip), out_fpath)

                '''
                masked_out_fpath = os.path.join(out_dir, 'sub-%s_space-%s_mask-%s_run-%02d_cond-%s_map-%s.nii.gz'%(subject_id, space_label, mask_descrip,  
                                                                                                                   rx, cond_label, stat_descrip))
                masked_img = masker.inverse_transform(fmri_masked)
                nib.save(masked_img, masked_out_fpath)
                '''
