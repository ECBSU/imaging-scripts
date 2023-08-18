#!/usr/bin/python3
import os
import numpy as np
from PIL import Image
import multiprocessing
import argparse 

#Input options
parser = argparse.ArgumentParser(description='Histogram -> contrast strecing -> 8-bit conversion.')
parser.add_argument('-in',dest ='input_folder', required = 'TRUE' , help='Input folder of 16-bit images')
parser.add_argument('-out', dest ='output_folder', required = 'TRUE' , help='Output folder of 8-bit images')
parser.add_argument('-t', dest ='template', required = 'TRUE' , help='Full path to template used for histogram matching')
parser.add_argument('-up', dest ='upper', required = 'TRUE' , type = int, help='Upper threshold for streatching')
parser.add_argument('-low', dest ='lower', required = 'TRUE' , type = int, help='Lower threshold for streatching')
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
template_image_path = args.template
upper = args.upper
lower = args.lower

#Allow for large images
Image.MAX_IMAGE_PIXELS = None
#Histogram of the template image
print("Calculating template histogram")
template_image = np.array(Image.open(template_image_path))
template = template_image.ravel()
t_values, t_counts = np.unique(template, return_counts=True)
t_cdf = np.cumsum(t_counts).astype(np.float64)
t_cdf /= t_cdf[-1]
print("Calculated template histogram")

# Compute the histogram of the source and template images
def histogram_match(source, file_name):
    print("Calculate histogram: " + file_name)
    oldshape = source.shape
    source = source.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.float64)
    s_cdf /= s_cdf[-1]
    matched = np.interp(s_cdf, t_cdf, t_values)
    
    return matched[bin_idx].reshape(oldshape)

def stretch(a, lower_thresh, upper_thresh):
    r = 65535.0/(upper_thresh-lower_thresh+2) # unit of stretching
    out = np.round(r*(a-lower_thresh+1)).astype(a.dtype) # stretched values
    out[a<lower_thresh] = 0
    out[a>upper_thresh] = 65535
    return out

#performs contrast streatching then convert images from 16 bit to 8 bit
def process_image(input_file, template_image):
    im = np.array(Image.open(input_file))
    matched_image = histogram_match(im, input_file)
    print("Matched histogram: " + input_file)
    stretch_im = stretch(matched_image, lower, upper)
    #convert to 8bit
    im_8bit = (stretch_im / stretch_im.max()) * 255.
    im_8bit = np.uint8(im_8bit)
    adjusted_image_pil = Image.fromarray(im_8bit)
    output_file = os.path.join(output_folder, os.path.basename(os.path.splitext(input_file)[0]) + "_adjusted.tif")
    adjusted_image_pil.save(output_file)
    print("Saved: " + output_file)

def main():
    input_files = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder) if filename.endswith('.tif')]
    num_processes = multiprocessing.cpu_count() 
    pool = multiprocessing.Pool(processes=num_processes)
    pool.starmap(process_image, [(input_file, template_image) for input_file in input_files])
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
