#!/bin/bash

fwhm=1.50

for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/derivatives/fmriprep_noSDC/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sbatch run_modeling_firstlevel_stimulus.sh $subid $fwhm
done
