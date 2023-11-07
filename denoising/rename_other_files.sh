#!/bin/bash
# after creating `_acq-dwidenoise` data,
# need to rename other files in the `func/` directory

data_dir=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/
func_dir=$data_dir/${1}/func/

# bold.json
for fpath in $func_dir/*bold.json; do
  echo $fpath
  fbase=$(basename $fpath)
  out_base="${fbase:0: -10}_acq-dwidenoise${fbase: -10}"
  out_path=$func_dir/$out_base
  echo $out_path
  mv $fpath $out_path
done

# sbref
for fpath in $func_dir/*sbref.nii.gz; do
  echo $fpath
  fbase=$(basename $fpath)
  out_base="${fbase:0: -13}_acq-dwidenoise${fbase: -13}"
  out_path=$func_dir/$out_base
  echo $out_path
  mv $fpath $out_path
done

# sbref.json
for fpath in $func_dir/*sbref.json; do
  echo $fpath
  fbase=$(basename $fpath)
  out_base="${fbase:0: -11}_acq-dwidenoise${fbase: -11}"
  out_path=$func_dir/$out_base
  echo $out_path
  mv $fpath $out_path
done

# events.tsv
for fpath in $func_dir/*events.tsv; do
  echo $fpath
  fbase=$(basename $fpath)
  out_base="${fbase:0: -11}_acq-dwidenoise${fbase: -11}"
  out_path=$func_dir/$out_base
  echo $out_path
  mv $fpath $out_path
done

