#!/bin/bash

fwhm=1.50

for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do 
  subid=$(basename $subpath)
  echo $subid
  sbatch run_modeling_firstlevel_denoised_stimulus-LSS.sh $subid
done
