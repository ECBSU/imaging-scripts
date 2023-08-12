import os
import numpy as np
from PIL import Image
import multiprocessing

Image.MAX_IMAGE_PIXELS = None
input_folder = r'D:\Uni\Dino\Projects\Chilostomela\dataset_fuse'
template_image_path = r'D:\Uni\Dino\Projects\Chilostomela\dataset_fuse\s0032_fused_tp_0_ch_0.tif'
template_image = np.array(Image.open(template_image_path))

# Compute the histogram of the source and template images
def histogram_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.float64)
    t_cdf = np.cumsum(t_counts).astype(np.float64)
    s_cdf /= s_cdf[-1]
    t_cdf /= t_cdf[-1]
    matched = np.interp(s_cdf, t_cdf, t_values)
    
    return matched[bin_idx].reshape(oldshape)

#convert images from 16 bit to 8 bit
def process_image(input_file, template_image):
    im = np.array(Image.open(input_file))
    matched_image = histogram_match(im, template_image)
    
    im_8bit = (matched_image / matched_image.max()) * 255.
    im_8bit = np.uint8(im_8bit)
    adjusted_image_pil = Image.fromarray(im_8bit)
    output_file = os.path.splitext(input_file)[0] + "_adjusted.tif"
    adjusted_image_pil.save(output_file)

#run parallel processing
def main():
    input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.tif')]
    num_processes = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(processes=num_processes)
    pool.starmap(process_image, [(input_file, template_image) for input_file in input_files])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
