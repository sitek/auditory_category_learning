#!/bin/bash
#SBATCH --mem=20G
#SBATCH --time=4:00:00

# convert dicoms to bids-standard niftis using heudiconv
# KRS 2022.02.16

module add dcm2niix

# define paths
data_dir=/bgfs/bchandrasekaran/krs228/data/FLT/
software_dir=/bgfs/bchandrasekaran/krs228/software/

sub=$1

## get the singularity image
#singularity pull docker://nipy/heudiconv:0.10.0 $software_dir/singularity_images/heudiconv

# run preliminary conversion to get dicom info table
echo "running heudiconv setup on $1"
heudiconv -d "${data_dir}/sourcedata/dicoms/{subject}/{subject}*/scans/*/resources/DICOM/files/*.dcm" \
  -s $sub \
  -c none -b \
  -ss 1 \
  -o $data_dir/data_bids \
  -f convertall 
