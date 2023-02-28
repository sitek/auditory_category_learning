#!/bin/bash
#SBATCH --time=6:00:00

# convert dicoms to bids-standard niftis using heudiconv
# THIS SCRIPT IS JUST FOR FLT25--manually transferred instead of XNAT
# KRS 2023.01.17

module add dcm2niix

# define paths
data_dir=/bgfs/bchandrasekaran/krs228/data/FLT/
software_dir=/bgfs/bchandrasekaran/krs228/software/

sub=FLT25

## get the singularity image
#singularity pull docker://nipy/heudiconv:0.10.0 $software_dir/singularity_images/heudiconv

# run  conversion
echo "converting $1"
heudiconv -d "${data_dir}/sourcedata/dicoms/{subject}/*/*/*" \
  -s $sub \
  -c dcm2niix \
  --bids \
  -ss 1 \
  --grouping all \
  -o $data_dir/data_bids_noIntendedFor \
  -f heuristic_noIntendedFor.py
