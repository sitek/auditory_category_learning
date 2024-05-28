#!/bin/bash
#SBATCH --time=16:00:00

# atlas options: 'dseg', 'subcort-aud'
# model options: 'stimulus_per_run_LSS', 'run-all_LSS'
for model in stimulus_per_run_LSS run-all_LSS; do
#model=run-all_LSS
#for atlas in dseg subcort-aud; do
atlas=tian-S3
python mask_trial_betas.py --sub=$1 \
                           --fwhm=0.00 \
                           --atlas=$atlas \
                           --space=MNI152NLin2009cAsym \
                           --stat=tstat \
                           --model=$model \
    --mask_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/derivatives/nilearn/masks/ \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/
#done
done



