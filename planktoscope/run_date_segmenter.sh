#!/bin/bash
# To run local_segmenter for planktoscope-acquired images, insert SD card into the Dell workstation and run directly.

# Function to display help message
display_help() {
    echo "Usage: $0 -d|--date <DATE>  -i|--in_dir <path/to/pkscope/data/img>  -o|--out_dir <output/path> "
}

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--date)
            date="$2"
            shift 2
            ;;
        -i|--in_dir)
            in="$2"
            shift 2
            ;;
        -o|--out_dir)
            outdir="$2"
            shift 2
            ;;
        -h|--help)
            display_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Check if the date variable was set
if [[ -z "$date" ]]; then
    echo "Error: No date provided."
    display_help
    exit 1
fi

# Check if input and output directories are set
if [[ -z "$in" ]]; then
    echo "Error: No input directory provided. Using default </media/ecbsu/rootfs/home/pi/data/img/$date> "
    in="/media/ecbsu/rootfs1/home/pi/data/img/$date"
fi

if [[ -z "$outdir" ]]; then
    echo "Error: No output directory provided. Using default </media/ecbsu/vEM_data/planktoscope_data> "
    outdir='/media/ecbsu/vEM_data/planktoscope_data'
fi

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

wait

for site in $(find "$in" -mindepth 1 -maxdepth 1 -type d); do
  o="$outdir/$date/$(basename "$site")"
  for run in "$site"/* ; do cp "$run/metadata.json" "$o/$(basename "$run")/metadata.json"; done
done 

# Check if the segmenter completed runs
echo "Completed runs:"
find $outdir/$date -maxdepth 4 -name done.txt 
