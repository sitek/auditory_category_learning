#!/bin/bash

fwhm=0

#for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do 
#  subid=$(basename $subpath)

for subid in FLT02 FLT03 FLT04 FLT05 FLT06 FLT07 FLT08 FLT09 FLT10 FLT11 FLT12 FLT13 FLT14 FLT15 FLT17 FLT18 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT26 FLT28 FLT30; do
  echo $subid
  sbatch run_mask_trial_betas_run-grouped.sh $subid
done
