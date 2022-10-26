#!/bin/bash

#SBATCH --time=3:00:00

#conda activate py3

python modeling_firstlevel_stimulus_perrun.py --sub=$1 --task=tonecat \
    --space=MNI152NLin2009cAsym --fwhm=0 --event_type=stimulus --t_acq=2 --t_r=3

