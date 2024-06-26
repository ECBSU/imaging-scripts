# Change to the date of run
date='20240612'

# to run local_segmenter for planktoscope
conda deactivate
conda deactivate
source /home/ecbsu/Desktop/seg_dir/seg_dir/bin/activate

# to run by date folders
outdir='/media/ecbsu/vEM_data/planktoscope_data'

in="/media/ecbsu/rootfs/home/pi/data/img/$date"
mkdir $outdir/$date

for site in $(find "$in" -mindepth 1 -maxdepth 1 -type d); do
  o="$outdir/$date/$(basename "$site")"
  mkdir -p $o
  for run in "$site"/* ; do python /home/ecbsu/Desktop/seg_dir/local_segmenter_cmd.py -i $run -o $o -c & done
done
