import torch
from models.my_svd_pipeline_clean import My_SVD as StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif
import os
from utils.nvs_solver_utils import search_hypers
from PIL import Image

from models.unet import UNetSpatioTemporalConditionModel
import numpy as np

def load_pipeline(checkpoint, use_nvs_solver=False):
    svd_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        svd_path,
        subfolder='unet',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
        device_map=None,
        use_safetensors=True,
        using_traj_attn=True,
    )

    it = 0
    param_list = torch.load(checkpoint)
    for name, para in unet.named_parameters():
        if 'trajectory' in name:
            para.requires_grad_(False)
            para.copy_(param_list[it])
            it += 1

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        svd_path, torch_dtype=torch.float16, variant="fp16",
        unet=unet,
        use_nvs_solver=use_nvs_solver
    )

    pipeline.enable_model_cpu_offload()
    return pipeline

def get_val_batch_traj(flow_path, first_frame, num_frames):

    traj_path = flow_path+"/trans_coordinates.npy"
    valid_mask = flow_path+"/trans_valid.npy"
    val_batch = dict()
    val_batch['trajectory'] = torch.from_numpy(np.load(traj_path)).reshape(num_frames,-1,2)
    val_batch['masks'] = torch.from_numpy(np.load(valid_mask).reshape(num_frames,-1))

    val_batch['first_frame'] = first_frame
    val_batch['height'] = 576
    val_batch['width'] = 1024

    return val_batch

def generate(pipeline, flow_path, image_path, output_folder, num_frames, 
             seed=12345, use_nvs_solver=False, nvs_solver_cond_path=None, num_inference_steps=25):
    val_batch = get_val_batch_traj(flow_path, first_frame=image_path, 
                                   num_frames=num_frames)

    image = load_image(val_batch['first_frame'])
    width = 1024
    height = 576
    image = image.resize((width, height))
    generator = torch.manual_seed(seed)
    pipeline.set_flow_path(trajectory=val_batch['trajectory'], occ_mask=val_batch['masks'],
                            height=val_batch['height'], width=val_batch['width'])

    if use_nvs_solver:
        cond_image = []
        masks = []
        for i in range(24):
            # cond_image.append(Image.open(os.path.join(save_path, 'fix', str(i).zfill(4)+'.png')))
            # mask_erosion = Image.open(os.path.join(save_path, 'fix', str(i).zfill(4)+'_mask.png'))
            cond_image.append(Image.open(os.path.join(nvs_solver_cond_path, str(i).zfill(4)+'.png')))
            mask_erosion = Image.open(os.path.join(nvs_solver_cond_path, str(i).zfill(4)+'_mask.png'))
            mask_erosion_ = np.array(mask_erosion)/255.
            mask_erosion_[mask_erosion_ < 0.5] = 0
            mask_erosion_[mask_erosion_ >= 0.5] = 1

            mask_erosion = np.mean(mask_erosion_,axis = -1)
            mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
            mask_erosion = np.mean(mask_erosion,axis=2)
            mask_erosion[mask_erosion < 0.2] = 0
            mask_erosion[mask_erosion >= 0.2] = 1
            masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))

        masks = torch.concat(masks)

        sigma_list = np.load(f'utils/sigmas/sigmas_{num_inference_steps}.npy').tolist()
        lambda_ts = search_hypers(sigma_list)
    else:
        masks = None
        lambda_ts = None
        cond_image = None



    frames = pipeline(image, decode_chunk_size=8, num_inference_steps=num_inference_steps, generator=generator, 
                    num_frames=num_frames, noise_aug_strength=0.02, 
                    height=height, width=width, motion_bucket_id=127, 
                    max_guidance_scale=3,
                    temp_cond=cond_image, 
                    mask=masks, 
                    lambda_ts=lambda_ts,
                        ).frames[0]

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"output_{seed}.gif")
    export_to_gif(frames, output_path)
    print(f"Genration finished, saved to {output_path}")