#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 4

python multivariate_roi.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --analysis_window=run \
    --fwhm=0.00 \
    --cond=assigned \
    --contrast=sound \
    --maptype=tstat \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/ \
    --mask_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/derivatives/nilearn/masks/

