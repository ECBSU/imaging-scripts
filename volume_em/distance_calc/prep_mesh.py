#!/usr/bin/env python3
import trimesh
import os
import sys

def split_objects(mesh):
    print("splitting objects")
    individual_objects = mesh.split(only_watertight=False)
    print(f"Found {len(individual_objects)} individual objects")
    return individual_objects

def bound_external(obj):
    # Create and combine a bounding box around the external object
    box = obj.bounding_box_oriented.to_mesh()
    combined = trimesh.util.concatenate([box, obj])
    return combined

def main():
    mesh_input = sys.argv[1]  # Path to the mesh input file

    tmp_path = "tmp_individual_objects"
    print(f"Temporary individual object directory created at: {tmp_path}")
    os.makedirs(tmp_path, exist_ok=True)
    
    mesh = trimesh.load(mesh_input)
    individual_objects = split_objects(mesh)
    print("writing individual objects")
    for individual_object in individual_objects:
        obj_id = id(individual_object)
        temp_filename = f"individual_object_{obj_id}.ply"
        individual_object.export(f"{tmp_path}/{temp_filename}")

if __name__ == "__main__":
    main()
