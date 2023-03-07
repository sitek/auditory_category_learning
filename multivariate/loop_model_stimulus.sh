#!/bin/bash

fwhm=1.50

for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/derivatives/22.1.1/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sbatch run_modeling_firstlevel_stimulus_perrun.sh $subid $fwhm
done
