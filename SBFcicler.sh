#!/bin/bash

#Specify the slice (z value) from where to continue/start the stitching of images, and the total of slices to be processed

slices=3546
begining=1034


for ((j = begining; j <= slices; j++)); do
        
        #Modify the imj macro with each loop--------------------------------------------------
        new_string="i = $j"
        
        #File to edit
        file="Bigstitcher_SBF_Teneo+bash.ijm"
        
        # Use sed to replace the first line with the new string
        sed -i "1s/.*/$new_string/" "$file"

        #Run the ImageJ macro ----------------------------------------------------------------

        #Path to the ImageJ executable and the macro script
        IMAGEJ_EXECUTABLE="/home/ecbsu/programs/Fiji.app/ImageJ-linux64"
        MACRO_SCRIPT="/media/ecbsu/vEM_data/Javier/Bigstitcher_SBF_Teneo+bash.ijm"
        
        #Run IMJ macro
        $IMAGEJ_EXECUTABLE -macro $MACRO_SCRIPT
        echo "Slice $j done------------------------------------------------------------------"

        #Clean the ram
        sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'

        
done
