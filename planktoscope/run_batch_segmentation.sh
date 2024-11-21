#!/bin/bash

# activate enviroment
conda deactivate
conda deactivate
source /home/ecbsu/Desktop/seg_dir/seg_dir/bin/activate

### Input parameters
outdir='/media/ecbsu/vEM_data/planktoscope_data'
segmentor='/home/ecbsu/Desktop/seg_dir/local_segmenter_cmd.py'

# To run all dates
for date in /media/ecbsu/rootfs/home/pi/data/img/*; do in="/media/ecbsu/rootfs/home/pi/data/img/$(basename "$date")"
  mkdir -p $outdir/$(basename "$date")
  for site in $(find "$in" -mindepth 1 -maxdepth 1 -type d); do
    o="$outdir/$(basename "$date")/$(basename "$site")"
    mkdir -p $o
    for run in "$site"/* ; do python $segmentor -i $run -o $o -c & 
    done
  done
done

wait
for date in /media/ecbsu/rootfs/home/pi/data/img/*; do in="/media/ecbsu/rootfs/home/pi/data/img/$(basename "$date")"
  for site in $(find "$in" -mindepth 1 -maxdepth 1 -type d); do
    o="$outdir/$(basename "$date")/$(basename "$site")"
    for run in "$site"/* ; do cp "$run/metadata.json" "$o/$(basename "$run")/metadata.json"; done
  done 
done

# Check if the segmenter completed runs
echo "Completed runs:"
find $outdir -maxdepth 4 -name done.txt 

