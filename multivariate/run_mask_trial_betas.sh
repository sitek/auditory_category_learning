#!/bin/bash
#SBATCH --time=4:00:00

# atlas options: 'dseg', 'subcort-aud'
python mask_trial_betas.py --sub=$1 \
                           --fwhm=0 \
                           --atlas=subcort-aud \
                           --space=MNI152NLin2009cAsym \
                           --stat=beta \
                           --model=stimulus_per_run

