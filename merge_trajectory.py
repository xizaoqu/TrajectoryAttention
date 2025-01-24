import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_trajectory_folder"
)

parser.add_argument(
    "--camera_conditioned_folder"
)

parser.add_argument(
    "--merged_trajectory_folder"
)

parser.add_argument('--num_frames',type=int,default=25) 

args = parser.parse_args()

video_trans_coordinates = torch.from_numpy(np.load(os.path.join(args.video_trajectory_folder, 'trans_coordinates.npy')))
video_valid = torch.from_numpy(np.load(os.path.join(args.video_trajectory_folder, 'trans_valid.npy')))

x_indices = video_trans_coordinates[..., 0].long()  # (F, N)
y_indices = video_trans_coordinates[..., 1].long()  # (F, N)
video_valid = video_valid & (x_indices>=0) & (x_indices<1024) & (y_indices>=0) & (y_indices<576)


camera_conditioned_trans_coordinates = torch.from_numpy(np.load(os.path.join(args.camera_conditioned_folder, 'trans_coordinates.npy')))
camera_conditioned_valid = torch.from_numpy(np.load(os.path.join(args.camera_conditioned_folder, 'trans_valid.npy')))

# import pdb;pdb.set_trace()
camera_conditioned_valid = camera_conditioned_valid[:,::4,::4].reshape(args.num_frames, -1)

num_frames = args.num_frames

# import pdb;pdb.set_trace()
for i in range(num_frames-1):

    flow = camera_conditioned_trans_coordinates[i+1] - camera_conditioned_trans_coordinates[0]
    x_indices_i = x_indices[0, i+1][video_valid[0, i+1]]
    y_indices_i = y_indices[0, i+1][video_valid[0, i+1]]

    offset = flow[y_indices_i, x_indices_i]
    
    video_trans_coordinates[0, i+1, video_valid[0, i+1]] += offset


x_indices = video_trans_coordinates[..., 0].long()  # (F, N)
y_indices = video_trans_coordinates[..., 1].long()  # (F, N)

merge_valid = video_valid & camera_conditioned_valid & (x_indices>=0) & (x_indices<1024) & (y_indices>=0) & (y_indices<576)

os.makedirs(args.merged_trajectory_folder, exist_ok=True)
np.save(os.path.join(args.merged_trajectory_folder, 'trans_coordinates.npy'), video_trans_coordinates.numpy())
np.save(os.path.join(args.merged_trajectory_folder, 'trans_valid.npy'), merge_valid.numpy())