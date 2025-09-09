#!/usr/bin/env python3
import trimesh
import os
import sys

def bound_external(obj):
    # Create and combine a bounding box around the external object
    box = obj.bounding_box_oriented.to_mesh()
    combined = trimesh.util.concatenate([box, obj])
    return combined

def main():
    external_object_path = sys.argv[1]  # Path to the external object file
    print("Generating bounded external")
    external_object = trimesh.load(external_object_path)
    external_object = bound_external(external_object)
    external_object.export("temp_external.ply")
    print("bounded external object exported to temp_external.ply")    

if __name__ == "__main__":
    main()
