#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -c 2

python multivariate_searchlight.py --sub=$1 --space=MNI152NLin2009cAsym --fwhm=1.5 --cond=tone
