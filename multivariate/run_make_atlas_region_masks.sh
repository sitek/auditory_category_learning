#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH -c 2

# atlas options: carpet_dseg, subcort_aud, tian_S3, carpet_motor
for sub in FLT02 FLT03 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT16 FLT17 FLT18 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
#for sub in FLT04; do
  python make_atlas_region_masks.py --sub=$sub \
    --space=MNI152NLin2009cAsym \
    --fwhm=0.00 \
    --atlas_label=carpet_motor \
    --bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_bids/ \
    --fmriprep_dir=/bgfs/bchandrasekaran/krs228/data/FLT/derivatives/22.1.1/
done
