// Define the size of subvolumes
subvolumeSizeX = 128;
subvolumeSizeY = 128;
subvolumeSizeZ = 128;
out_dir="D:/Uni/unet_test/unet/train/images/Substack"

// Get the active 3D image
getDimensions(width, height, channels, slices, frames);
img=getTitle()
// Calculate the number of subvolumes in each dimension
numSubvolumesX = width / subvolumeSizeX;
numSubvolumesY = height / subvolumeSizeY;
numSubvolumesZ = slices / subvolumeSizeZ;

// Iterate through each subvolume
for (i = 0; i < numSubvolumesX; i++) {
    for (j = 0; j < numSubvolumesY; j++) {
        xStart = i * subvolumeSizeX;
        yStart = j * subvolumeSizeY;
        xEnd = xStart + subvolumeSizeX - 1;
        yEnd = yStart + subvolumeSizeY - 1;
        makeRectangle(xStart, yStart, subvolumeSizeX, subvolumeSizeY);
        for (k = 0; k < numSubvolumesZ; k++) {
            // Define the ROI for the subvolume
            zStart = k * subvolumeSizeZ;
            zEnd = zStart + subvolumeSizeZ - 1;
            run("Make Substack...", "slices=" + (zStart + 1) + "-" + (zEnd + 1));
            saveAs("Tiff", out_dir + "/" + img + (xStart + 1) + "_" + (yStart + 1) + "_" + (zStart + 1) + ".tif");
            run("Close");
            selectImage(img);
        }
    }
}
