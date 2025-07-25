//set path to working dir,  this should be the parent of the folder containing the images. In windows use a double '\' (eg. \\path\\to\\your\\folder\\)
work_dir = "/media/ecbsu/vEM_data/Phua/psolani/" 
//set name of folder containing the images
image_folder = "crop"
//x,y = tiles; z = total slices; i=starting slice
x = 3
y = 3 
z = 1015
i = 16

//downsampling settings
pairwise_shift_dwnsample = 4
output_dwn_sample = 2 //output are in 16-bit; minis are downsampled by 8 and in 8-bit

dataset_dir = work_dir+"dataset" //if images are not stitched properly, this contains the files to check, s****.h5 can be opened in bigstitcher
output_dir = work_dir+"dataset_fuse"
//make intermediate folders
File.makeDirectory(dataset_dir); 
File.makeDirectory(output_dir);
File.makeDirectory(output_dir + "/mini");
//run loop
z = z+1
for (i = i; i < z; i++) {
print("Generating dataset for s"+String.pad(i,4));

//Run bigstitcher creates dataset.xml and dataset.h5, for each z stack in the folder, stitch 3x3 tiles together.
run("Define Multi-View Dataset", "define_dataset=[Automatic Loader (Bioformats based)] project_filename=s"+String.pad(i,4)+".xml path="+work_dir+image_folder+"/*"+String.pad(i,4)+"_e01.tif exclude=10 pattern_0=Tiles pattern_1=Tiles modify_voxel_size? voxel_size_x=20 voxel_size_y=20 voxel_size_z=65 voxel_size_unit=nm move_tiles_to_grid_(per_angle)?=[Move Tile to Grid (Macro-scriptable)] grid_type=[Right & Down             ] tiles_x="+x+" tiles_y="+y+" tiles_z=1 overlap_x_(%)=10 overlap_y_(%)=10 overlap_z_(%)=10 keep_metadata_rotation how_to_store_input_images=[Re-save as multiresolution HDF5] metadata_save_path="+work_dir+"dataset/ image_data_save_path="+work_dir+"dataset/ check_stack_sizes subsampling_factors=[{ {1,1,1}, {2,2,1}, {4,4,1}, {8,8,1}, {16,16,2}, {32,32,4} }] hdf5_chunk_sizes=[{ {64,64,1}, {64,64,1}, {64,64,1}, {64,64,1}, {64,64,1}, {64,64,1} }] timepoints_per_partition=1 setups_per_partition=0 use_deflate_compression");

//checks if dataset is generated for alignement.
if (File.exists( work_dir+"dataset" + "/s"+String.pad(i,4)+".xml")){//checks if dataset is generated for alignement.
print("Starting alignmentr for s"+String.pad(i,4));
//alignment of images
run("Calculate pairwise shifts ...", "select="+ dataset_dir + "/s"+String.pad(i,4)+".xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] method=[Phase Correlation] downsample_in_x=" + pairwise_shift_dwnsample + " downsample_in_y=" + pairwise_shift_dwnsample + "");
run("Filter pairwise shifts ...", "select="+ dataset_dir + "/s"+String.pad(i,4)+".xml min_r=0 max_r=1 max_shift_in_x=0 max_shift_in_y=0 max_shift_in_z=0 max_displacement=0");
run("Optimize globally and apply shifts ...", "select="+ dataset_dir + "/s"+String.pad(i,4)+".xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500 global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] fix_group_0-0");
//stitch and output tif file, 16 bit
run("Image Fusion", "select="+ dataset_dir + "/s"+String.pad(i,4)+".xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=" + output_dwn_sample + " interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] define_input=[Auto-load from input data (values shown below)] output_file_directory="+output_dir + " filename_addition=[s"+String.pad(i,4)+"]");

run("Image Fusion", "select="+ dataset_dir + "/s"+String.pad(i,4)+".xml process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] process_tile=[All tiles] process_timepoint=[All Timepoints] bounding_box=[Currently Selected Views] downsampling=16 interpolation=[Linear Interpolation] pixel_type=[16-bit unsigned integer] interest_points_for_non_rigid=[-= Disable Non-Rigid =-] blend produce=[Each timepoint & channel] fused_image=[Save as (compressed) TIFF stacks] define_input=[Auto-load from input data (values shown below)] output_file_directory="+output_dir + "/mini filename_addition=[mini_s"+String.pad(i,4)+"]");
}
else{
	print("Error: "+ dataset_dir + "/s"+String.pad(i,4)+".xml is not found/generated.");
	break;
}
}
