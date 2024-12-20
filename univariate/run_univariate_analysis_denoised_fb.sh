#!/bin/bash

#SBATCH --time=12:00:00

python univariate_analysis_fb-correct-vs-wrong.py --sub=$1 \
                              --task=tonecat \
                              --space=MNI152NLin2009cAsym \
                              --fwhm=0.00 \
                              --event_type=stimulus \
                              --t_acq=2 --t_r=3 \
                              --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/ \
                              --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/derivatives/denoised_fmriprep-22.1.1/

