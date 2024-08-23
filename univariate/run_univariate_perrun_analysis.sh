#!/bin/bash

#SBATCH --time=3:00:00

python univariate_perrun_analysis.py --sub=$1 \
                              --task=tonecat \
                              --space=MNI152NLin2009cAsym \
                              --fwhm=3 \
                              --event_type=sound \
                              --t_acq=2 --t_r=3 \
                              --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
                              --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/
