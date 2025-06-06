import os
import numpy as np
import nibabel as nib

from nilearn.glm.contrasts import compute_fixed_effects
from glob import glob

runcombo_dict = {'early': ['run00', 'run01'], 
                 'middle': ['run02', 'run03'], 
                 'final': ['run04','run05']}

deriv_dir = os.path.join('/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/',
                         'derivatives/nilearn/bids-deriv_level-1_fwhm-0.00')

space_label = 'MNI152NLin2009cAsym'
model_label = 'per_run_LSS_confound-compcor_event-stimulus' # 'LSA'
'''
for sub_label in ['FLT02', 'FLT03', 'FLT04', 'FLT05', 'FLT06', 'FLT07', 'FLT08',
                  'FLT09', 'FLT10', 'FLT11', 'FLT12', 'FLT13', 'FLT14', 'FLT15',
                  'FLT17', 'FLT18', 'FLT19', 'FLT20', 'FLT21', 'FLT22',
                  'FLT23', 'FLT24', 'FLT25', 'FLT26', 'FLT28', 'FLT30']:
'''
for sub_label in ['FLT15']:
    sub_dir = os.path.join(deriv_dir, 
                           f'sub-{sub_label}_space-{space_label}',
                           model_label,)
    print(sub_label)

    out_sub_dir = os.path.join(sub_dir, 'run-grouped')
    os.makedirs(out_sub_dir, exist_ok=True)                          

    # get contrast labels from filenames
    stat_maps = sorted(glob(sub_dir+f'/*/sub*/*Di*stat-effect_statmap.nii.gz')) 
    conditions_all = sorted(set([os.path.basename(x).split('_')[4] for x in (stat_maps)]))

    conditions_all

    for rc in runcombo_dict:
        print(rc)
        run_list = runcombo_dict[rc]

        for cond_label in conditions_all:
            print(cond_label)

            contrast_imgs = [sorted(glob(sub_dir+f'/{run_id}/*/*{cond_label}_stat-effect_statmap.nii.gz'))[0]
                             for run_id in run_list]
            variance_imgs = [sorted(glob(sub_dir+f'/{run_id}/*/*{cond_label}_stat-variance_statmap.nii.gz'))[0]
                             for run_id in run_list]

            fixed_fx_contrast, fixed_fx_variance, \
                               fixed_fx_stat, \
                               fixed_fx_z_score = compute_fixed_effects(contrast_imgs,
                                                                     variance_imgs,
                                                                     return_z_score=True)

            out_variance_fpath = os.path.join(out_sub_dir, 
                                          f'sub-{sub_label}_rungroup-{rc}_{cond_label}_stat-variance_statmap.nii.gz')
            nib.save(fixed_fx_variance, out_variance_fpath)

            out_stat_fpath = os.path.join(out_sub_dir, 
                                          f'sub-{sub_label}_rungroup-{rc}_{cond_label}_stat-t_statmap.nii.gz')
            nib.save(fixed_fx_stat, out_stat_fpath)