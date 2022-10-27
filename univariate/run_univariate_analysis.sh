#!/bin/bash

#SBATCH --time=3:00:00

python univariate_analysis.py --sub=$1 --task=tonecat \
                              --space=MNI152NLin2009cAsym \
                              --fwhm=4.5 --event_type=sound \
                              --t_acq=2 --t_r=3