#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -c 2

#conda activate py3

python modeling_firstlevel_singleevent.py --sub=$1 --task=tonecat \
    --space=MNI152NLin2009cAsym --fwhm=0 \
    --event_type=stimulus --t_acq=2 --t_r=3 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/archive/data_bids_noIntendedFor/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/
