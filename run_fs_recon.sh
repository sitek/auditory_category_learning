#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --time=1-00

project_dir=/bgfs/bchandrasekaran/krs228/data/FLT/

subjid=FLT01

subjectsdir=$project_dir/derivatives/freesurfer
mkdir -p $subjectsdir

in_file=${project_dir}/data_bids/sub-${subjid}/anat/sub-${subjid}_T1w.nii.gz

recon-all -all -s $subjid -i $in_file -sd $subjectsdir
