#!/bin/bash

#SBATCH --time=23:00:00
#SBATCH --mem=32G

#conda activate py3

bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/

python modeling_firstlevel_stimulus_perrun.py --sub=$1 --task=tonecat \
    --space=MNI152NLin2009cAsym --fwhm=0 \
    --event_type=stimulus --model_type=LSS \
    --t_acq=2 --t_r=3 \
    --bidsroot=$bidsroot \
    --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-22.1.1/
