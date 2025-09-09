#!/bin/bash
# input variables
mesh="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/syntro_20250902_legendrea_smol_stack_filter_full.ply"
external_mesh="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/cell_exterior_20250728_cell_Lables_legendrea_smol_stack_filter_full.ply"
output_file="syntro_20250902_dist.tsv"
num_vertices=500 # Maximum number of vertices used. Precision of calculation
cpu_count=2

# enable (true) of disable (false) mesh prep
prep_external=false
prep_mesh=true

# check if input/debug files exist
if [ -f "$output_file" ]; then
    echo "File $output_file already exists."
    exit 1
fi
if [ -f "killed_processes.log" ]; then
    echo "File killed_processes.log already exists."
    exit 1
fi

# prepare meshes for calculation
if [ "$prep_mesh" = true ]; then
    python prep_mesh.py $mesh $external_mesh &
fi

if [ "$prep_external" = true ]; then
    python prep_external.py $external_mesh &
fi
wait

# run distance calculations
touch "$output_file"
find tmp_individual_objects -name "*.ply" | xargs --max-args=1 --max-procs=$cpu_count -I {} sh -c "
    python calc_dist.py {} $num_vertices $output_file || { echo 'Killed: {}' >> killed_processes.log; mkdir -p failed_objects; cp {} failed_objects/; }
"
wait

# check for failed runs, rerun them serially
if [ -f "killed_processes.log" ]; then
    echo "Some processes were killed. Objects in killed_processes.log will be rerun serially."
    echo 'Re-run of failed processes' >> failed_processes.log
    find failed_objects -name "*.ply" | xargs --max-args=1 -I {} sh -c "
        python calc_dist.py {} $num_vertices $output_file || echo 'Failed: {}' >> failed_processes.log
    "
fi
