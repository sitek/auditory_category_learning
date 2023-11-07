#!/bin/bash
#SBATCH --time=23:00:00

module load mrtrix3

data_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/
func_dir=$data_dir/${1}/func/

for func_path in $func_dir/*.nii.gz; do
  echo $func_path
  func_base=$(basename $func_path)
  echo "denoising $func_base"
  out_base="${func_base:0: -12}_acq-dwidenoise${func_base: -12}"
  echo "will save to $out_base"
  dwidenoise $func_path $func_dir/$out_base -info -debug
done
