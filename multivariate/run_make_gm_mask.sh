#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -c 1

python make_gm_mask.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --fwhm=1.5 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids_noIntendedFor/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/