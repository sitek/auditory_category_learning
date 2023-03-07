#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 2

python multivariate_permuted_searchlight.py \
    --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --fwhm=1.5 \
    --n_permutations=100 \
    --searchrad=$2 \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/archive/data_bids_noIntendedFor/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/

