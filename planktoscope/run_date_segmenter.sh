# To run local_segmenter for planktoscope-acquired images, insert SD card into the Dell workstation and run directly.

# This will select the date that the segmenter will run on.
date='20240612'

# Output and input dir information
outdir='/media/ecbsu/vEM_data/planktoscope_data'
in="/media/ecbsu/rootfs/home/pi/data/img/$date"

# Deactivating conda and activating segmenter environment
conda deactivate
conda deactivate
source /home/ecbsu/Desktop/seg_dir/seg_dir/bin/activate

# Make output folder
mkdir $outdir/$date

# Run segmenter on all sites and runs
for site in $(find "$in" -mindepth 1 -maxdepth 1 -type d); do
  o="$outdir/$date/$(basename "$site")"
  mkdir -p $o
  for run in "$site"/* ; do python /home/ecbsu/Desktop/seg_dir/local_segmenter_cmd.py -i $run -o $o -c & done # the -c flag here is used to remove debug files (extremely large files)
done
