ITEM=ride
CAMERA_MOTION_MODE=horizontal
DEGREE=-0.5

set -e

# depth estimation
python Depth-Anything-V2/run.py \
  --encoder vitl \
  --img-path data/${ITEM}/images  \
  --outdir data/${ITEM}/depth

# trajectory extraction
python trajectory_extraction.py \
  --image_folder data/${ITEM}/images/ \
  --depth_folder data/${ITEM}/depth/ \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
  --degrees_per_frame ${DEGREE} \
  --camera_motion_mode ${CAMERA_MOTION_MODE} \
  --major_radius 200 \
  --num_frames 25

# generaiton
python generate.py \
  --image_folder data/${ITEM}/images \
  --trajectory_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE}/trajectories \
  --num_frames 25 \
  --output_folder output/${ITEM}/${CAMERA_MOTION_MODE}_${DEGREE} \
  --checkpoint checkpoints/trajattn_temp.pth 