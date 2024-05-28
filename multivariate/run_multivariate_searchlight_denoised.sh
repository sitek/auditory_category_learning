#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -c 4
#SBATCH --mem=64G

python multivariate_searchlight.py --sub=$1 \
    --space=MNI152NLin2009cAsym \
    --analysis_window=session \
    --fwhm=0.00 \
    --cond=assigned \
    --contrast=fb \
    --searchrad=4.5 \
    --maptype=tstat \
    --mask_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/derivatives/nilearn/masks/ \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/

