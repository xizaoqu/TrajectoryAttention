from diffusers import StableVideoDiffusionPipeline
from diffusers.models import AutoencoderKLTemporalDecoder
from models.unet import UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch
import numpy as np
from diffusers.models.attention_processor import (
    XFormersAttnProcessor, AttnProcessor2_0, Attention, deprecate
)
from typing import Optional
import xformers
from diffusers.utils.torch_utils import randn_tensor
import PIL
from typing import Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims, StableVideoDiffusionPipelineOutput
from PIL import Image
import torch.nn.functional as F
from einops import rearrange

def tensor2vid(video: torch.Tensor, processor, output_type="np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs

def random_unique_samples(N, M):
    all_values = torch.arange(N)
    perm = torch.randperm(N)
    selected_values = all_values[perm[:M]]
    
    return selected_values

def bool_tensor_to_gif(tensor, gif_path, duration=100):
    """
    Convert a F×H×W boolean tensor to a GIF.
    
    Parameters:
    tensor (np.ndarray): A boolean numpy array of shape (F, H, W).
    gif_path (str): The path where the output GIF will be saved.
    duration (int): Duration of each frame in milliseconds.
    """
    # Convert boolean tensor to uint8 (0 or 255) for image representation
    frames = (tensor.cpu().numpy() * 255).astype(np.uint8)
    
    # Create a list to store frames as images
    image_frames = [Image.fromarray(frame) for frame in frames]
    
    # Save frames as a GIF
    image_frames[0].save(
        gif_path,
        save_all=True,
        append_images=image_frames[1:],
        duration=duration,
        loop=0
    )

class VanillaAttnProcessor1_0(XFormersAttnProcessor): # to mute warning
    
    def __init__(self, name=None, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        self.name=name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        height=None,
        width=None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        # if 'up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1' in self.name:
        #     torch.save(hidden_states.reshape(2,height,width, hidden_states.shape[-2], hidden_states.shape[-1]).cpu(), "cal_query_key_sim/hidden_states.pt")

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)

        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class VanillaAttnProcessor2_0(AttnProcessor2_0): # to mute warning
    
    def __init__(self, name=None, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        self.name=name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        height=None,
        width=None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class MotionVectorAttnProcessor1_0(XFormersAttnProcessor):
    def __init__(self, trajectory, occ_mask, origin_size, sparsity_factor, attention_op=None,
                 name=None):
        super().__init__(attention_op)
        self.occ_mask=occ_mask
        self.trajectory = trajectory
        self.name = name
        self.origin_size = origin_size
        self.sparsity_factor = sparsity_factor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        height = None,
        width = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        total_, frame_number, channel_number = hidden_states.shape
        hidden_states = hidden_states.reshape(int(total_/height/width), height, width, frame_number, channel_number)

        # downsample = max(int(self.origin_size[1]/height), 1)
        # trajectory = self.trajectory.to(hidden_states.device).reshape(12,self.origin_size[1],self.origin_size[0],2)
        # trajectory = trajectory[:,::downsample,::downsample]
        # trajectory = trajectory.reshape(12, -1, 2)
        # valid_mask = self.occ_mask.to(hidden_states.device).reshape(12,self.origin_size[1],self.origin_size[0]).bool()
        # valid_mask = valid_mask[:,::downsample,::downsample]
        # valid_mask = valid_mask.reshape(12, -1)

        # sample
        traj_sample_num = int(width*height/self.sparsity_factor)
        traj_sample_idx = random_unique_samples(self.trajectory.shape[1], traj_sample_num*32)
        trajectory = self.trajectory.to(hidden_states.device)[:,traj_sample_idx]
        trajectory[..., 0] *= width / self.origin_size[0]
        trajectory[..., 1] *= height / self.origin_size[1]
        valid_mask = self.occ_mask.to(hidden_states.device)[:,traj_sample_idx].bool()

        x_indices = trajectory[..., 0].long()  # (F, N)
        y_indices = trajectory[..., 1].long()  # (F, N)
        x_indices = x_indices.clamp(0, width-1)
        y_indices = y_indices.clamp(0, height-1)

        _, N, _ = trajectory.shape
        frame_indices = torch.arange(frame_number).unsqueeze(1).expand(frame_number, N).to(hidden_states.device)
        indices = torch.stack((frame_indices, y_indices, x_indices), dim=-1)  # (F, N, 3)
        traj_hidden_states = hidden_states[:, indices[..., 1], indices[..., 2], indices[..., 0], :]
        traj_hidden_states = traj_hidden_states * valid_mask[None, :, :, None]
        traj_hidden_states = traj_hidden_states.permute(0,2,1,3)
        hidden_states = traj_hidden_states.reshape(-1, frame_number, channel_number)

        # do attention
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, op=self.attention_op, scale=attn.scale)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # project back
        mapped_features = torch.zeros((int(total_/height/width), frame_number, height, width, channel_number), 
                                        dtype=hidden_states.dtype,
                                        device=hidden_states.device)
        count = torch.zeros((int(total_/height/width), frame_number, height, width), 
                            dtype=hidden_states.dtype,
                            device=hidden_states.device)

        hidden_states = hidden_states.reshape(int(total_/height/width), -1, frame_number, channel_number)
        
        for f in range(frame_number):
            x_valid = indices[...,2][f][valid_mask[f]]
            y_valid = indices[...,1][f][valid_mask[f]]

            for bb in range(mapped_features.shape[0]):
                mapped_features[bb,f].index_put_((y_valid, x_valid), hidden_states[bb,valid_mask[f],f], accumulate=True)
                count[bb,f].index_put_((y_valid, x_valid), hidden_states[bb,valid_mask[f],f,0]*0+1, accumulate=True)
        nonzero_mask = count > 0
        mapped_features[nonzero_mask] /= count[nonzero_mask][..., None]
        hidden_states = mapped_features
        hidden_states = hidden_states.permute(0,2,3,1,4).reshape(total_, frame_number, channel_number)

        return hidden_states

class MotionVectorAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self, trajectory, occ_mask, origin_size, sparsity_factor, attention_op=None,
                 name=None):
        self.occ_mask=occ_mask
        self.trajectory = trajectory
        self.name = name
        self.origin_size = origin_size
        self.sparsity_factor = sparsity_factor

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        height = None,
        width = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        total_, frame_number, channel_number = hidden_states.shape
        hidden_states = hidden_states.reshape(int(total_/height/width), height, width, frame_number, channel_number)

        # downsample = max(int(self.origin_size[1]/height), 1)
        # trajectory = self.trajectory.to(hidden_states.device).reshape(12,self.origin_size[1],self.origin_size[0],2)
        # trajectory = trajectory[:,::downsample,::downsample]
        # trajectory = trajectory.reshape(12, -1, 2)
        # valid_mask = self.occ_mask.to(hidden_states.device).reshape(12,self.origin_size[1],self.origin_size[0]).bool()
        # valid_mask = valid_mask[:,::downsample,::downsample]
        # valid_mask = valid_mask.reshape(12, -1)

        # sample
        traj_sample_num = int(width*height/self.sparsity_factor)
        traj_sample_idx = random_unique_samples(self.trajectory.shape[1], traj_sample_num*8)
        trajectory = self.trajectory.to(hidden_states.device)[:,traj_sample_idx]
        trajectory[..., 0] *= width / self.origin_size[0]
        trajectory[..., 1] *= height / self.origin_size[1]
        valid_mask = self.occ_mask.to(hidden_states.device)[:,traj_sample_idx].bool()

        x_indices = trajectory[..., 0].long()  # (F, N)
        y_indices = trajectory[..., 1].long()  # (F, N)
        x_indices = x_indices.clamp(0, width-1)
        y_indices = y_indices.clamp(0, height-1)

        _, N, _ = trajectory.shape
        frame_indices = torch.arange(frame_number).unsqueeze(1).expand(frame_number, N).to(hidden_states.device)
        indices = torch.stack((frame_indices, y_indices, x_indices), dim=-1)  # (F, N, 3)
        traj_hidden_states = hidden_states[:, indices[..., 1], indices[..., 2], indices[..., 0], :]
        traj_hidden_states = traj_hidden_states * valid_mask[None, :, :, None]
        traj_hidden_states = traj_hidden_states.permute(0,2,1,3)
        hidden_states = traj_hidden_states.reshape(-1, frame_number, channel_number)

        # do attention
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        hidden_states = hidden_states / attn.rescale_output_factor
        
        # project back
        mapped_features = torch.zeros((int(total_/height/width), frame_number, height, width, channel_number), 
                                        dtype=hidden_states.dtype,
                                        device=hidden_states.device)
        count = torch.zeros((int(total_/height/width), frame_number, height, width), 
                            dtype=hidden_states.dtype,
                            device=hidden_states.device)

        hidden_states = hidden_states.reshape(int(total_/height/width), -1, frame_number, channel_number)
        
        for f in range(frame_number):
            x_valid = indices[...,2][f][valid_mask[f]]
            y_valid = indices[...,1][f][valid_mask[f]]

            for bb in range(mapped_features.shape[0]):
                mapped_features[bb,f].index_put_((y_valid, x_valid), hidden_states[bb,valid_mask[f],f], accumulate=True)
                count[bb,f].index_put_((y_valid, x_valid), hidden_states[bb,valid_mask[f],f,0]*0+1, accumulate=True)
        nonzero_mask = count > 0
        mapped_features[nonzero_mask] /= count[nonzero_mask][..., None]
        hidden_states = mapped_features
        hidden_states = hidden_states.permute(0,2,3,1,4).reshape(total_, frame_number, channel_number)

        return hidden_states


if torch.__version__.startswith("2."):
    VanillaAttnProcessor = VanillaAttnProcessor2_0
    MotionVectorAttnProcessor = MotionVectorAttnProcessor2_0
else:
    VanillaAttnProcessor = VanillaAttnProcessor1_0
    MotionVectorAttnProcessor = MotionVectorAttnProcessor1_0
    
def set_temporal_attn(unet, trajectory, occ_mask, height, width,
                      multi_gpu=False):
    attn_processor_dict = {}

    if multi_gpu:
        for k in unet.module.attn_processors.keys():
            if "trajectory" in k:
                attn_processor_dict[k] = MotionVectorAttnProcessor(trajectory=trajectory, 
                                                                occ_mask=occ_mask, origin_size=(width,height), sparsity_factor=4, name=k) # MotionVectorAttnProcessor()
            else:
                attn_processor_dict[k] = VanillaAttnProcessor(name=k)
        unet.module.set_attn_processor(attn_processor_dict)
    else:
        for k in unet.attn_processors.keys():
            if "trajectory" in k:
                attn_processor_dict[k] = MotionVectorAttnProcessor(trajectory=trajectory, 
                                                                occ_mask=occ_mask, origin_size=(width,height), sparsity_factor=4, name=k) # MotionVectorAttnProcessor()
            else:
                attn_processor_dict[k] = VanillaAttnProcessor(name=k)
        unet.set_attn_processor(attn_processor_dict)

class My_SVD(StableVideoDiffusionPipeline):
    def __init__(self, vae: AutoencoderKLTemporalDecoder, image_encoder: CLIPVisionModelWithProjection, unet: UNetSpatioTemporalConditionModel, 
                 scheduler: EulerDiscreteScheduler, feature_extractor: CLIPImageProcessor, 
                 use_nvs_solver = False):
        self.use_nvs_solver = use_nvs_solver
        if self.use_nvs_solver:
            from utils.nvs_solver_schedule import EulerDiscreteScheduler
            scheduler = EulerDiscreteScheduler(
                beta_end=scheduler.beta_end,
                beta_schedule=scheduler.beta_schedule,
                beta_start=scheduler.beta_start,
                interpolation_type=scheduler.interpolation_type,
                num_train_timesteps=scheduler.num_train_timesteps,
                prediction_type=scheduler.prediction_type,
                rescale_betas_zero_snr=scheduler.rescale_betas_zero_snr,
                sigma_max=scheduler.sigma_max,
                sigma_min=scheduler.sigma_min,
                steps_offset=scheduler.steps_offset,
                timestep_spacing=scheduler.timestep_spacing,
                timestep_type=scheduler.timestep_type,
                trained_betas=scheduler.trained_betas,
                use_karras_sigmas=scheduler.use_karras_sigmas,                
            )
        super().__init__(vae, image_encoder, unet, scheduler, feature_extractor)
        

    def set_flow_path(self, trajectory, occ_mask, height, width):
        attn_processor_dict = {}
        for k in self.unet.attn_processors.keys():
            if "trajectory" in k:
                attn_processor_dict[k] = MotionVectorAttnProcessor(trajectory=trajectory, 
                                                                occ_mask=occ_mask, origin_size=(width,height), sparsity_factor=4, name=k) # MotionVectorAttnProcessor()
            else:
                attn_processor_dict[k] = VanillaAttnProcessor(name=k)
        self.unet.set_attn_processor(attn_processor_dict)

    def __call__(
        self,
        image,
        temp_cond=None,
        mask=None,
        lambda_ts=None,
        lr=0.02,
        weight_clamp=0.6,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0, 1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames`
                (14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the expense of more memory usage. By default, the decoder decodes all frames at once for maximal
                quality. For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.FloatTensor`) is returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        with torch.no_grad():
            image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)

        if self.use_nvs_solver:
            mask = mask.cuda()
            mask = mask.unsqueeze(1).unsqueeze(0).repeat(1,1,4,1,1)
            temp_cond_list = []
            for i in range(len(temp_cond)):
                temp_cond_ = self.image_processor.preprocess(temp_cond[i], height=height, width=width)
                temp_cond_list.append(temp_cond_)
            temp_cond = torch.cat(temp_cond_list,dim=0)

        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        with torch.no_grad():
            image_latents = self._encode_vae_image(
                image,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )

        image_latents = image_latents.to(image_embeddings.dtype)
        image_latents = rearrange(image_latents, "(b f) c h w -> b f c h w",f=1)
        if self.use_nvs_solver:
            with torch.no_grad():
                temp_cond_latents_list = []
                for i in range(temp_cond.shape[0]):
                    temp_cond_latents_ = self._encode_vae_image(temp_cond[i:i+1,:,:,:], device, 
                                                                num_videos_per_prompt, self.do_classifier_free_guidance) # [12, 4, 72, 128]
                    temp_cond_latents_ = rearrange(temp_cond_latents_, "(b f) c h w -> b f c h w",b=2)
                    temp_cond_latents_list.append(temp_cond_latents_)
            temp_cond_latents = torch.cat(temp_cond_latents_list,dim=1)
           
            temp_cond_latents  = torch.cat((image_latents,temp_cond_latents),dim=1)

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.repeat(1, num_frames, 1, 1, 1)

        batch, num_frames, channels, h, w = image_latents.shape
     
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if self.use_nvs_solver:
                    for g in range(1):
                        grads = []
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input_test1=latent_model_input
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t,step_i=i)
                        latent_model_input_test2=latent_model_input
                        latent_model_input = torch.cat([latent_model_input[0:1], image_latents[1:2]], dim=2)
                        latent_model_input2 = latent_model_input
                        for ii in range(4):
                            with torch.enable_grad(): 
                            
                                latents.requires_grad_(True)
                                latents.retain_grad()
                                image_latents.requires_grad_(True)         
                                latent_model_input = latent_model_input.detach()
                                latent_model_input.requires_grad = True

                                named_param = list(self.unet.named_parameters())
                                for n,p in named_param:
                                    p.requires_grad = False
                                if ii == 0:
                                    latent_model_input1 = latent_model_input[0:1,:,:,:40,:72]
                                    latents1 = latents[0:1,:,:,:40,:72]
                                    temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,:72]
                                    mask1 = mask[0:1,:,:,:40,:72]
                                elif ii ==1:
                                    latent_model_input1 = latent_model_input[0:1,:,:,24:,:72]
                                    latents1 = latents[0:1,:,:,24:,:72]
                                    temp_cond_latents1 = temp_cond_latents[:2,:,:,24:,:72]
                                    mask1 = mask[0:1,:,:,24:,:72]
                                elif ii ==2:
                                    latent_model_input1 = latent_model_input[0:1,:,:,:40,56:]
                                    latents1 = latents[0:1,:,:,:40,56:]
                                    temp_cond_latents1 = temp_cond_latents[:2,:,:,:40,56:]
                                    mask1 = mask[0:1,:,:,:40,56:]
                                elif ii ==3:
                                    latent_model_input1 = latent_model_input[0:1,:,:,24:,56:]
                                    latents1 = latents[0:1,:,:,24:,56:]
                                    temp_cond_latents1 = temp_cond_latents[:2,:,:,24:,56:]
                                    mask1 = mask[0:1,:,:,24:,56:]
                                image_embeddings1 = image_embeddings[0:1,:,:]
                                added_time_ids1 =added_time_ids[0:1,:]
                                torch.cuda.empty_cache()
                                noise_pred_t = self.unet(
                                    latent_model_input1,
                                    t,
                                    encoder_hidden_states=image_embeddings1,
                                    added_time_ids=added_time_ids1,
                                    return_dict=False,
                                    use_traj=False
                                )[0]
                            
                                output = self.scheduler.step_single(noise_pred_t, t, latents1,temp_cond_latents1,mask1,lambda_ts,step_i=i,lr=lr,weight_clamp=weight_clamp,compute_grad=True)
                                grad = output.grad
                                grads.append(grad)
                                
                        grads1 = torch.cat((grads[0],grads[1][:,:,:,16:,:]),-2)                
                        grads2 = torch.cat((grads[2],grads[3][:,:,:,16:,:]),-2)
                        grads3 = torch.cat((grads1,grads2[:,:,:,:,16:]),-1)
                        latents = latents - grads3.half()

                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    
                    if self.use_nvs_solver:
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t, step_i=i)
                    else:
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                        use_traj = True
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if self.use_nvs_solver:
                        latents = self.scheduler.step_single(noise_pred, t, latents,temp_cond_latents,mask,lambda_ts,step_i=i,compute_grad=False).prev_sample
                    else:
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        if not output_type == "latent":
            
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            with torch.no_grad():
                frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()


        if not return_dict:
            return frames
        
        return StableVideoDiffusionPipelineOutput(frames=frames)
