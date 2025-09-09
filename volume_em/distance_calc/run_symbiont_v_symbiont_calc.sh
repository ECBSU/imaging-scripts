#!/bin/bash
cpu_count=2
num_vertices=500

# input ply
methanobacterium="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/methanobacterium_20250902_cell_Lables_legendrea_smol_stack_filter_full.ply"
methanosaeta="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/methanosaeta_20250902_legendrea_smol_stack_filter_full.ply"
syntro="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/syntro_20250902_legendrea_smol_stack_filter_full.ply"
cox="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/cox_20250728_cell_Lables_legendrea_smol_stack_filter_full.ply"
input_ply=("$methanobacterium" "$methanosaeta" "$syntro" "$cox")

# splited obj dir
methanobacterium_split="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/dist_calc_script/methanobacterium_individual_objects/"
methanosaeta_split="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/dist_calc_script/methanosaeta_individual_objects/"
syntro_split="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/dist_calc_script/syntro_individual_objects/"
cox_split="/media/ecbsu/vEM_data/Phua/Legend/smol_stack_filter/dist_calc_script/cox_individual_objects/"
split_object=("$methanobacterium_split" "$methanosaeta_split" "$syntro_split" "$cox_split")

# calculate distance between symbionts
for i in "${input_ply[@]}"; do
    i_label=$(basename "$i" | cut -d'_' -f1) 
    for x in "${split_object[@]}"; do
        x_label=$(basename "$x" | cut -d'_' -f1)
        # skip if they match
        if [[ "$i_label" == "$x_label" ]]; then
            continue
        fi
        output_file="$(basename "$i_label" .ply)_dist_frm_${x_label}.txt"
        find $x -name "*.ply" | xargs --max-args=1 --max-procs=$cpu_count -I {} sh -c "
        python calc_symbiont_distance.py {} $num_vertices $output_file $i || echo 'Killed: {}' >> ${output_file}_killed_processes.log
"
    done
done
