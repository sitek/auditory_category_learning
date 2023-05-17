#!/bin/bash

#SBATCH --time=6:00:00

python task-stgrid_univariate.py --sub=$1 --task=stgrid \
  --space=MNI152NLin2009cAsym --fwhm=3 \
  --event_type=block_stim --t_acq=2 --t_r=4 \
  --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/ \
  --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/22.1.1/