ITEM=gold-fish

set -e

# trajectory extraction
python cotracker.py \
    --video_path data/${ITEM}/${ITEM}.mp4 \
    --output_path output/${ITEM}/trajectories

# generaiton
python generate.py \
  --image_folder data/${ITEM}/editing_frame \
  --trajectory_folder output/${ITEM}/trajectories \
  --num_frames 25 \
  --output_folder output/${ITEM}/ \
  --checkpoint checkpoints/trajattn_temp.pth