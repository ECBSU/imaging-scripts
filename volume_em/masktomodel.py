import numpy as np
import pyvista as pv
import tifffile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input TIFF file')
parser.add_argument('--fmt', default='stl', help='Output format (e.g., stl, ply, obj)')
parser.add_argument('-x', '--x', type=float, required=True, help='pix size of x in um')
parser.add_argument('-y', '--y', type=float, required=True, help='pix size of y in um')
parser.add_argument('-z', '--z', type=float, required=True, help='pix size of z in um')
parser.add_argument('--resize', type=float, help='Resize factor')
parser.add_argument('--deci', type=float, help='Decimation factor, reduce complexity of model (greatly influences filesize). Fraction of the original mesh to remove. If target_reduction is set to 0.9, this filter will try to reduce the data set to 10-percnet of its original size and will remove 90-percent of the input triangles.')
args = parser.parse_args()

if args.deci is not None:
    if args.deci < 0.0 or args.deci > 1.0:
        raise argparse.ArgumentTypeError(f"{args.deci} not in range [0.0, 1.0]")
    
print("Loading image", args.input)
volume = tifffile.imread(args.input)
print("Mask to contour")
grid = pv.wrap(volume.astype(np.uint8))
contours = grid.contour(isosurfaces=[0.5])
print("Cleaning mesh")
contours = contours.clean(progress_bar=True)

print("Scaling XYZ for 1:1:1 aspect ratio")
min_size = min(args.x, args.y, args.z)
#Note that the scaling is ZXY, different from the manual where they use xyz. tif mask slices from MIB should be in pos 1
scaling_factors = [args.z / min_size, args.y / min_size, args.x / min_size]
contours.scale(scaling_factors[0], scaling_factors[1], scaling_factors[2])

if args.deci is not None:
    print("Decimate vol by", args.deci)
    contours = contours.decimate_pro(args.deci, progress_bar=True)
    contours = contours.clean(progress_bar=True)  
if args.resize is not None:
    print("Rescaling vol by", args.resize)
    contours = contours.scale([args.resize, args.resize, args.resize])

print("Saving file:", args.input.replace(".tif", f".{args.fmt}"))
contours.save(args.input.replace(".tif", f".{args.fmt}"))
