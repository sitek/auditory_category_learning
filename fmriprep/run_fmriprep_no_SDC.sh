#!/bin/bash
#SBATCH --time=1-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Preprocess single-subject FLT data using fmriprep
# in a Singularity container
# KRS 2022.02.15

module add freesurfer
module add fsl
module add afni
module add ants
module add singularity/3.8.3

#conda activate py3

# define paths
software_path=/bgfs/bchandrasekaran/krs228/software/
project_path=/bgfs/bchandrasekaran/krs228/data/FLT/
data_dir=$project_path/data_bids_noIntendedFor/

analysis_desc=fmriprep_noSDC/
work_dir=/bgfs/bchandrasekaran/krs228/work/${analysis_desc}
out_dir=$project_path/derivatives/${analysis_desc}/

# singularity
sing_dir=$software_path/singularity_images/
sing_img=$sing_dir/fmriprep-22.0.0.simg

# define inputs
fs_license=$software_path/license.txt
sub=$1

# previously run freesurfer outputs
fs_subjects_dir=${project_path}/derivatives/fmriprep/sourcedata/freesurfer/

# copy from SBATCH arguments
mem=64000
nprocs=4
omp_n=2

# BEFORE RUNNING FOR THE FIRST TIME: 
# build the fmriprep container to a singularity image
# (will only build from head node; no unsquashfs when running from nodes)
#singularity build $sing_img docker://nipreps/fmriprep:21.0.1

#--fs-subjects-dir $fs_subjects_dir \
# run fmriprep
singularity run --cleanenv -B /bgfs:/bgfs $sing_img \
  $data_dir $out_dir participant \
  --participant-label $sub \
  --fs-license-file $fs_license \
  --work-dir $work_dir \
  --skip_bids_validation \
  --fs-subjects-dir $fs_subjects_dir \
  -vv \
  --mem $mem \
  --nprocs $nprocs --omp-nthreads $omp_n \
  --output-spaces T1w fsnative MNI152NLin2009cAsym

