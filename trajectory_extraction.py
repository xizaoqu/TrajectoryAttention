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

parser.add_argument(
    "--control_mode",
    type=str,
    default='image'
)

args = parser.parse_args()

image_path = os.listdir(args.image_folder)
image_path.sort()
depth_path = os.listdir(args.depth_folder)
depth_path.sort()

depth_path = [d for d in depth_path if d.endswith('.npy')]
assert len(image_path) == len(depth_path)

image_path = [os.path.join(args.image_folder, ip) for ip in image_path]
depth_path = [os.path.join(args.depth_folder, dp)for dp in depth_path]

if args.control_mode == 'image':
    image_path = image_path[0:1]
    depth_path = depth_path[0:1]

# for image camera control
if len(image_path) == 1:
    image_path = image_path * args.num_frames
    depth_path = depth_path * args.num_frames

cameractrl(image_path, depth_path, args.output_folder, args.num_frames,
           major_radius=args.major_radius, minor_radius=200, degrees_per_frame=args.degrees_per_frame, camera_motion_mode=args.camera_motion_mode)