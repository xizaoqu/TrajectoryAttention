from utils.generate import load_pipeline, generate
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_folder",
    type=str
)

parser.add_argument(
    "--output_folder",
    type=str
)

parser.add_argument(
    "--trajectory_folder",
    type=str
)

parser.add_argument(
    "--num_frames",
    type=int
)

parser.add_argument(
    "--seed",
    type=int,
    default=12345
)

parser.add_argument(
    "--checkpoint",
    type=str,
)

parser.add_argument(
    "--use_nvs_solver",
    type=str,
)

parser.add_argument(
    "--nvs_solver_cond_path",
    type=str,
)

parser.add_argument(
    "--num_inference_steps",
    type=int,
    default=25
)

args = parser.parse_args()

checkpoint = args.checkpoint
pipeline = load_pipeline(checkpoint, use_nvs_solver=args.use_nvs_solver)

image_path = os.listdir(args.image_folder)
image_path.sort()
image_path = os.path.join(args.image_folder, image_path[0])

generate(pipeline, args.trajectory_folder, image_path, 
         args.output_folder, args.num_frames, seed=args.seed,
         use_nvs_solver=args.use_nvs_solver,
         nvs_solver_cond_path=args.nvs_solver_cond_path,
         num_inference_steps=args.num_inference_steps)