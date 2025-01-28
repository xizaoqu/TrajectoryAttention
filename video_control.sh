ITEM=train
CAMERA_MOTION_MODE=horizontal
DEGREE=-0.8

set -e

python extract_frames.py \
    --video_path data/${ITEM}/${ITEM}.mp4 \
    --output_folder data/${ITEM}/images

# depth estimation
python Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images  \
  --outdir data/${ITEM}/depth

# camera-conditioned trajectory extraction
python trajectory_extraction.py \
  --image_folder data/${ITEM}/images/ \
  --depth_folder data/${ITEM}/depth/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 200 \
  --num_frames 25 \
  --control_mode video

# video trajectory extraction
python cotracker.py \
    --video_path data/${ITEM}/${ITEM}.mp4 \
    --output_path output/${ITEM}/trajectories \

python merge_trajectory.py \
  --video_trajectory_folder output/${ITEM}/trajectories \
  --camera_conditioned_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
  --merged_trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}_merged/trajectories

# generaiton
python generate.py \
  --image_folder data/${ITEM}/images \
  --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}_merged/trajectories \
  --num_frames 25 \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}_merged50step \
  --checkpoint checkpoints/trajattn_temp.pth \
  --use_nvs_solver \
  --nvs_solver_cond_path output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
  --num_inference_steps 50

# # # generaiton wo nvs
# python generate.py \
#   --image_folder data/${ITEM}/images \
#   --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}_merged/trajectories \
#   --num_frames 25 \
#   --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}_merged_100step_wonvs \
#   --checkpoint checkpoints/trajattn_temp.pth \
#   --nvs_solver_cond_path output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
#   --num_inference_steps 25
