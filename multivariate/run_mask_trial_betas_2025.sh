#!/bin/bash
#SBATCH --time=16:00:00

# atlas options: 'tian-S3', 'dseg', 'subcort-aud'
# model options: 'stimulus_per_run_LSS', 'run-all_LSS'
bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/

#for model in stimulus_per_run_LSS run-all_LSS; do
#model=per_run_LSA_confound-compcor_event-stimulus
model=per_run_LSA_confound-compcor_event-stimulus

for atlas in tian-S3 dseg subcort-aud; do
#atlas=tian-S3
python mask_trial_betas_2025.py --sub=$1 \
                           --fwhm=0.00 \
                           --atlas=$atlas \
                           --space=MNI152NLin2009cAsym \
                           --stat=t \
                           --model=$model \
                           --window=grouped-runs \
                            --mask_dir=$bidsroot/derivatives/nilearn/masks/ \
                            --bidsroot=$bidsroot \
                            --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-22.1.1/
#done
done



