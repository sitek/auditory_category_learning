#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 2

python multivariate_searchlight.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --fwhm=1.5 \
    --cond=tone \
    --searchrad=4.5 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids_noIntendedFor/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/
