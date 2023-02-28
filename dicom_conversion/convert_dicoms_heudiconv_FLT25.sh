#!/bin/bash
#SBATCH --time=4:00:00

# convert dicoms to bids-standard niftis using heudiconv
# SEPARATE COMMAND FOR FLT25 - standard xnat transfer
# failed, so copied manually (and has different folder structure)
# KRS 2023.01.23

module add dcm2niix

# define paths
data_dir=/bgfs/bchandrasekaran/krs228/data/FLT/
software_dir=/bgfs/bchandrasekaran/krs228/software/

sub=FLT25

## get the singularity image
#singularity pull docker://nipy/heudiconv:0.11.6 $software_dir/singularity_images/heudiconv

# run  conversion
echo "converting $1"
#heudiconv -d "${data_dir}/sourcedata/dicoms/{subject}/*/scans/*/resources/DICOM/files/*" \
heudiconv -d "${data_dir}/sourcedata/dicoms/{subject}/*/*/*" \
  -s $sub \
  -c dcm2niix \
  --bids \
  --grouping all \
  -ss 1 \
  -o $data_dir/data_bids \
  -f heuristic.py
