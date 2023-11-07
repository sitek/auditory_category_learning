# accidentally had an `=` at the beginning of file names
# ran this from .../data_denoised/ folder
for fpath in sub-FLT*/func/=*; do 
  basen=$(basename $fpath)
  dirn=$(dirname $fpath)
  newbase=${basen:1}
  newfpath="$dirn/$newbase"
  mv $fpath $newfpath
done