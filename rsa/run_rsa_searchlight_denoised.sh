#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=32GB

python rsa_searchlight.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --analysis_window=rungroup \
    --fwhm=0.00 \
    --searchrad=5 \
    --contrast=sound \
    --mask_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/nilearn/masks/ \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/


