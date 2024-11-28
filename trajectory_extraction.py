from utils.cameractrl_ import cameractrl
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_folder",
    type=str
)
parser.add_argument(
    "--depth_folder",
    type=str
)

parser.add_argument(
    "--output_folder",
    type=str
)

parser.add_argument(
    "--degrees_per_frame",
    type=float
)

parser.add_argument(
    "--camera_motion_mode",
    type=str,
    choices=['horizontal', 'vertical', 'zoomin', 'zoomout'],
)

parser.add_argument(
    "--major_radius",
    type=int,
    default=200
)

parser.add_argument(
    "--num_frames",
    type=int,
    default=25
)

args = parser.parse_args()

image_path = os.listdir(args.image_folder)
depth_path = os.listdir(args.depth_folder)

depth_path = [d for d in depth_path if d.endswith('.npy')]

assert len(image_path) == len(depth_path)
assert len(image_path) == 1

image_path = os.path.join(args.image_folder, image_path[0])
depth_path = os.path.join(args.depth_folder, depth_path[0])

cameractrl(image_path, depth_path, args.output_folder, args.num_frames,
           major_radius=args.major_radius, minor_radius=200, degrees_per_frame=args.degrees_per_frame, camera_motion_mode=args.camera_motion_mode)