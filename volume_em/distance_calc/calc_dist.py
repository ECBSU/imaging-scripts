#!/usr/bin/env python3
import trimesh
import numpy as np
import os
import sys

def calc_obj_dist(individual_object, bounded_ext_obj, num_vertices):
    individual_object = trimesh.load(individual_object)
    if len(individual_object.vertices) > num_vertices:
        sampled_points = individual_object.sample(num_vertices)
    else:
        sampled_points = individual_object.vertices
    bounded_ext_obj = trimesh.load(bounded_ext_obj)
    distances_to_external = trimesh.proximity.closest_point(bounded_ext_obj, sampled_points)
    min_dist = np.min(distances_to_external[1])
    return min_dist

def main():
    individual_obj = sys.argv[1]  # Path to the individual object file
    num_vertices = int(sys.argv[2])
    output_path = sys.argv[3]

    external_object_path = "temp_external.ply"

    # Calculate distances in batches
    print(f"Calculating distances for {individual_obj}")
    min_dists = calc_obj_dist(individual_obj, external_object_path, num_vertices)

    with open(output_path, "a") as out:
        out.write(f"{os.path.basename(sys.argv[1])}\t{min_dists}\n")
    print(f"Distances written to {output_path}")

if __name__ == "__main__":
    main()
