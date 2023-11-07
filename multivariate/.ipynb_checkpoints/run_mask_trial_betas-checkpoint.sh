#!/bin/bash
#SBATCH --time=2:00:00

# atlas options: 'dseg', 'subcort-aud'
# model options: 'stimulus_per_run_LSS', 'run-all_LSS'
python mask_trial_betas.py --sub=$1 \
                           --fwhm=0.00 \
                           --atlas=subcort-aud \
                           --space=MNI152NLin2009cAsym \
                           --stat=tstat \
                           --model=stimulus_per_run_LSS

