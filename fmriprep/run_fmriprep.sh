#!/bin/bash
#SBATCH --time=3-00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12

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
data_dir=$project_path/data_bids/

analysis_desc=fmriprep
work_dir=/bgfs/bchandrasekaran/krs228/work/${analysis_desc}
out_dir=$project_path/derivatives/${analysis_desc}/

# singularity
sing_dir=$software_path/singularity_images/
sing_img=$sing_dir/fmriprep-20.0.1.simg

# define inputs
fs_license=$software_path/license.txt
sub=$1

# previously run freesurfer outputs
#fs_subjects_dir=${project_path}/derivatives/fmriprep_2022.03.14/sourcedata/freesurfer/

# copy from SBATCH arguments
mem=100000
nprocs=12
omp_n=4

# BEFORE RUNNING FOR THE FIRST TIME: 
# build the fmriprep container to a singularity image
# (will only build from head node; no unsquashfs when running from nodes)
#singularity build $sing_img docker://nipreps/fmriprep:21.0.1

# run fmriprep
singularity run --cleanenv -B /bgfs:/bgfs $sing_img \
  $data_dir $out_dir participant \
  --participant-label $sub \
  --fs-license-file $fs_license \
  --work-dir $work_dir \
  --skip_bids_validation \
  -vv \
  --mem $mem \
  --nprocs $nprocs --omp-nthreads $omp_n \
  --output-spaces T1w func fsnative MNI152NLin2009cAsym

