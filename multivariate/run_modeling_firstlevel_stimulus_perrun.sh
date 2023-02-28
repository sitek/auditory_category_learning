#!/bin/bash

#SBATCH --time=6:00:00

#conda activate py3

python modeling_firstlevel_stimulus_perrun.py --sub=$1 --task=tonecat \
    --space=MNI152NLin2009cAsym --fwhm=1.5 \
    --event_type=stimulus --t_acq=2 --t_r=3 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids_noIntendedFor/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/

