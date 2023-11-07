#!/bin/bash
#SBATCH --time=3-00
#SBATCH --mem=54G
#SBATCH --cpus-per-task=8

# Preprocess single-subject FLT data using fmriprep
# in a Singularity container
# Updated to fmriprep 22.1.1 to deal with fieldmaps
# taking too long to process
# KRS 2023.01.20

module add freesurfer
module add fsl
module add afni
module add ants
module add singularity/3.8.3

#conda activate py3

# define paths
software_path=/bgfs/bchandrasekaran/krs228/software/
project_path=/bgfs/bchandrasekaran/krs228/data/FLT/
data_dir=$project_path/data_denoised/

fmriprep_version=22.1.1
analysis_desc="denoised_fmriprep-$fmriprep_version"
work_dir=/bgfs/bchandrasekaran/krs228/work/${analysis_desc}
out_dir=$data_dir/derivatives/${analysis_desc}/

# singularity
sing_dir=$software_path/singularity_images/
sing_img=$sing_dir/${fmriprep_version}.simg

# define inputs
fs_license=$software_path/license.txt
sub=$1

# previously run freesurfer outputs
fs_subjects_dir=${project_path}/derivatives/22.1.1/sourcedata/freesurfer/

# copy from SBATCH arguments
mem=64000
nprocs=8
omp_n=4

# BEFORE RUNNING FOR THE FIRST TIME: 
# build the fmriprep container to a singularity image
# (will only build from head node; no unsquashfs when running from nodes)
#singularity build $sing_img docker://nipreps/fmriprep:22.1.1

# run fmriprep
singularity run --cleanenv -B /bgfs:/bgfs $sing_img \
  $data_dir $out_dir participant \
  --participant-label $sub \
  --fs-license-file $fs_license \
  --fs-subjects-dir $fs_subjects_dir \
  --work-dir $work_dir \
  --skip_bids_validation \
  -vv \
  --mem $mem \
  --nprocs $nprocs --omp-nthreads $omp_n \
  --output-spaces T1w func fsnative MNI152NLin2009cAsym

