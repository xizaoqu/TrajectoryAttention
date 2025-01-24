import torch
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video_path",
    type=str
)
parser.add_argument(
    "--output_path",
    type=str
)
parser.add_argument(
    "--grid_size",
    type=int
)

args = parser.parse_args()

import imageio.v3 as iio
frames = iio.imread(args.video_path, plugin="FFMPEG")  # plugin="pyav"

device = 'cuda'
grid_size = args.grid_size
video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker2").to(device)

F, W, H = 25, 1024, 576  # Frames, Width, Height
interval = 8
B = 1  # Batch size

pred_tracks = []
pred_visibilitys = []
for _i in range(4): # random sample 4 times
    x_coords = np.arange(0, W, interval)  
    y_coords = np.arange(0, H, interval)  
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing="ij")

    x_noise = np.random.randint(-interval//2, interval//2, size=x_grid.shape)
    y_noise = np.random.randint(-interval//2, interval//2, size=y_grid.shape)

    x_grid = x_grid + x_noise
    y_grid = y_grid + y_noise

    x_grid = x_grid.clip(0, W-1)
    y_grid = y_grid.clip(0, H-1)

    x_grid = x_grid.flatten()  
    y_grid = y_grid.flatten()

    queries = np.stack([x_grid*0, x_grid, y_grid], axis=-1)
    queries = torch.tensor(queries).float().to(device)
    pred_track, pred_visibility = cotracker(video, queries=queries[None])
    pred_tracks.append(pred_track)
    pred_visibilitys.append(pred_visibility)

pred_tracks = torch.cat(pred_tracks, dim=2)
pred_visibilitys = torch.cat(pred_visibilitys, dim=2)

os.makedirs(args.output_path, exist_ok=True)
np.save(os.path.join(args.output_path, "trans_coordinates.npy"), pred_tracks.cpu().numpy())
np.save(os.path.join(args.output_path, "trans_valid.npy"), pred_visibilitys.cpu().numpy())
