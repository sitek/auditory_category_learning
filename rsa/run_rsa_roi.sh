#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH -c 2

python rsa_roi.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --analysis_window=session \
    --fwhm=0.00 \
    --maptype=tstat \
    --mask_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/derivatives/nilearn/masks/ \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/


