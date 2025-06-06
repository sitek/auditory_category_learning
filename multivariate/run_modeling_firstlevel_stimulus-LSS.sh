#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH -c 2

#conda activate py3

python modeling_firstlevel_stimulus.py --sub=$1 --task=tonecat \
    --space=MNI152NLin2009cAsym --fwhm=1.50 \
    --event_type=stimulus --model_type=LSS \
    --t_acq=2 --t_r=3 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/22.1.1/
