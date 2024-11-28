# codebase based on https://github.com/ZHU-Zhiyu/NVS_Solver

import numpy as np
import PIL
import os
from typing import Tuple, Optional

import numpy as np
from typing import Optional
import os
import numpy as np
import PIL.Image 
from numba import njit

@njit
def mask_occlusion_traj(depth, trans_depth, trans_coordinates, trans_valid):
    depth_new = trans_depth * 0 + 10086
    h, w = depth.shape

    for hh in range(1,h):
        for ww in range(1, w):
            x1, y1 = trans_coordinates[hh-1,ww-1]
            x2, y2 = trans_coordinates[hh,ww]
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            if x1 < w and x1 >= 0 and x2 < w and x2 >= 0 and \
                y1 < h and y1 >= 0 and y2 < h and y2 >= 0 and \
                abs((depth[hh,ww] - depth[hh-1,ww-1])/depth[hh,ww]/depth[hh-1,ww-1])<0.1:
                    value_array = np.full(depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)].shape, trans_depth[hh,ww])
                    depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)] = np.minimum(value_array, depth_new[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)])
                    
    for hh in range(1,h):
        for ww in range(1, w):
            x1, y1 = trans_coordinates[hh,ww]
            x1 = int(x1)
            y1 = int(y1)

            if x1 < w and x1 >= 0 and y1 < h and y1 >= 0:
                if (trans_depth[hh, ww] - depth_new[y1, x1]>0.6):
                    trans_valid[hh, ww] = False
    
    return trans_valid

def forward_warp(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                    transformation1: np.ndarray, transformation2: np.ndarray, intrinsic1: np.ndarray,
                    intrinsic2: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                np.ndarray]:
    """
    Given a frame1 and global transformations transformation1 and transformation2, warps frame1 to next view using
    bilinear splatting.
    :param frame1: (h, w, 3) uint8 np array
    :param mask1: (h, w) bool np array. Wherever mask1 is False, those pixels are ignored while warping. Optional
    :param depth1: (h, w) float np array.
    :param transformation1: (4, 4) extrinsic transformation matrix of first view: [R, t; 0, 1]
    :param transformation2: (4, 4) extrinsic transformation matrix of second view: [R, t; 0, 1]
    :param intrinsic1: (3, 3) camera intrinsic matrix
    :param intrinsic2: (3, 3) camera intrinsic matrix. Optional
    """
    h, w = frame1.shape[:2]
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    assert frame1.shape == (h, w, 3)
    assert mask1.shape == (h, w)
    assert depth1.shape == (h, w)
    assert transformation1.shape == (4, 4)
    assert transformation2.shape == (4, 4)
    assert intrinsic1.shape == (3, 3)
    assert intrinsic2.shape == (3, 3)

    trans_points1, world_points = compute_transformed_points(depth1, transformation1, transformation2, intrinsic1,
                                                    intrinsic2)
    
    trans_coordinates = trans_points1[:, :, :2, 0] / (trans_points1[:, :, 2:3, 0])
    
    trans_depth1 = trans_points1[:, :, 2, 0]

    grid = create_grid(h, w)
    flow12 = trans_coordinates - grid
    flow12 = flow12.reshape(-1,2)
    flow12[trans_depth1.reshape(-1)<0] = 100000 # important
    flow12 = flow12.reshape(576,1024,2)

    trans_coordinates = trans_coordinates.reshape(-1,2)
    trans_coordinates[trans_depth1.reshape(-1)<0] = 100000
    trans_coordinates = trans_coordinates.reshape(576,1024,2)

    trans_valid = (trans_depth1>0)
    trans_valid = mask_occlusion_traj(depth1, trans_depth1, trans_coordinates, trans_valid)

    # warped_frame2, mask2 = bilinear_splatting(frame1, mask1, trans_depth1, flow12, None, is_image=True)
    warped_frame2, mask2 = None, None

    return warped_frame2, mask2,flow12, trans_coordinates, trans_valid

def compute_transformed_points(depth1: np.ndarray, transformation1: np.ndarray,
                                transformation2: np.ndarray, intrinsic1: np.ndarray,
                                intrinsic2: Optional[np.ndarray]):
    """
    Computes transformed position for each pixel location
    """
    h, w = depth1.shape
    if intrinsic2 is None:
        intrinsic2 = np.copy(intrinsic1)
    
    transformation = np.matmul(transformation2, np.linalg.inv(transformation1))

    y1d = np.array(range(h))
    x1d = np.array(range(w))
    x2d, y2d = np.meshgrid(x1d, y1d)
    ones_2d = np.ones(shape=(h, w))
    ones_4d = ones_2d[:, :, None, None]
    pos_vectors_homo = np.stack([x2d, y2d, ones_2d], axis=2)[:, :, :, None]

    intrinsic1_inv = np.linalg.inv(intrinsic1)
    intrinsic1_inv_4d = intrinsic1_inv[None, None]
    intrinsic2_4d = intrinsic2[None, None]
    depth_4d = depth1[:, :, None, None]
    trans_4d = transformation[None, None]


    unnormalized_pos = np.matmul(intrinsic1_inv_4d, pos_vectors_homo)
    world_points = depth_4d * unnormalized_pos
    world_points_homo = np.concatenate([world_points, ones_4d], axis=2)
    trans_world_homo = np.matmul(trans_4d, world_points_homo)
    trans_world = trans_world_homo[:, :, :3]
    trans_norm_points = np.matmul(intrinsic2_4d, trans_world)

    return trans_norm_points,world_points

def bilinear_splatting(frame1: np.ndarray, mask1: Optional[np.ndarray], depth1: np.ndarray,
                        flow12: np.ndarray, flow12_mask: Optional[np.ndarray], is_image: bool = False) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Using inverse bilinear interpolation based splatting
    :param frame1: (h, w, c)
    :param mask1: (h, w): True if known and False if unknown. Optional
    :param depth1: (h, w)
    :param flow12: (h, w, 2)
    :param flow12_mask: (h, w): True if valid and False if invalid. Optional
    :param is_image: If true, the return array will be clipped to be in the range [0, 255] and type-casted to uint8
    :return: warped_frame2: (h, w, c)
                mask2: (h, w): True if known and False if unknown
    """
    h, w, c = frame1.shape
    if mask1 is None:
        mask1 = np.ones(shape=(h, w), dtype=bool)
    if flow12_mask is None:
        flow12_mask = np.ones(shape=(h, w), dtype=bool)
    grid = create_grid(h, w)
    trans_pos = flow12 + grid

    trans_pos_offset = trans_pos + 1
    trans_pos_floor = np.floor(trans_pos_offset).astype('int')
    trans_pos_ceil = np.ceil(trans_pos_offset).astype('int')
    trans_pos_offset[:, :, 0] = np.clip(trans_pos_offset[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_offset[:, :, 1] = np.clip(trans_pos_offset[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_floor[:, :, 0] = np.clip(trans_pos_floor[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_floor[:, :, 1] = np.clip(trans_pos_floor[:, :, 1], a_min=0, a_max=h + 1)
    trans_pos_ceil[:, :, 0] = np.clip(trans_pos_ceil[:, :, 0], a_min=0, a_max=w + 1)
    trans_pos_ceil[:, :, 1] = np.clip(trans_pos_ceil[:, :, 1], a_min=0, a_max=h + 1)

    prox_weight_nw = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_sw = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_offset[:, :, 0] - trans_pos_floor[:, :, 0]))
    prox_weight_ne = (1 - (trans_pos_offset[:, :, 1] - trans_pos_floor[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))
    prox_weight_se = (1 - (trans_pos_ceil[:, :, 1] - trans_pos_offset[:, :, 1])) * \
                        (1 - (trans_pos_ceil[:, :, 0] - trans_pos_offset[:, :, 0]))

    sat_depth1 = np.clip(depth1, a_min=0, a_max=5000)
    log_depth1 = np.log(1 + sat_depth1)
    depth_weights = np.exp(log_depth1 / log_depth1.max() * 50)

    weight_nw = prox_weight_nw * mask1 * flow12_mask / depth_weights
    weight_sw = prox_weight_sw * mask1 * flow12_mask / depth_weights
    weight_ne = prox_weight_ne * mask1 * flow12_mask / depth_weights
    weight_se = prox_weight_se * mask1 * flow12_mask / depth_weights

    weight_nw_3d = weight_nw[:, :, None]
    weight_sw_3d = weight_sw[:, :, None]
    weight_ne_3d = weight_ne[:, :, None]
    weight_se_3d = weight_se[:, :, None]

    warped_image = np.zeros(shape=(h + 2, w + 2, c), dtype=np.float64)
    warped_weights = np.zeros(shape=(h + 2, w + 2), dtype=np.float64)

    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_nw_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), frame1 * weight_sw_3d)
    np.add.at(warped_image, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_ne_3d)
    np.add.at(warped_image, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), frame1 * weight_se_3d)

    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_floor[:, :, 0]), weight_nw)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_floor[:, :, 0]), weight_sw)
    np.add.at(warped_weights, (trans_pos_floor[:, :, 1], trans_pos_ceil[:, :, 0]), weight_ne)
    np.add.at(warped_weights, (trans_pos_ceil[:, :, 1], trans_pos_ceil[:, :, 0]), weight_se)

    cropped_warped_image = warped_image[1:-1, 1:-1]
    cropped_weights = warped_weights[1:-1, 1:-1]

    mask = cropped_weights > 0 
    mask2 = cropped_weights <=0.6
    mask = mask*mask2
    with np.errstate(invalid='ignore'):
        warped_frame2 = np.where(mask[:, :, None], cropped_warped_image / cropped_weights[:, :, None], 0)

    if is_image:
        assert np.min(warped_frame2) >= 0
        assert np.max(warped_frame2) <= 256
        clipped_image = np.clip(warped_frame2, a_min=0, a_max=255)
        warped_frame2 = np.round(clipped_image).astype('uint8')
    return warped_frame2, mask

def create_grid(h, w):
    x_1d = np.arange(0, w)[None]
    y_1d = np.arange(0, h)[:, None]
    x_2d = np.repeat(x_1d, repeats=h, axis=0)
    y_2d = np.repeat(y_1d, repeats=w, axis=1)
    grid = np.stack([x_2d, y_2d], axis=2)
    return grid

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def look_at_matrix(camera_position, target, up):

    # Camera's forward vector (z-axis)
    forward = normalize(target - camera_position)
    # Camera's right vector (x-axis)
    right = normalize(np.cross(up, forward))
    # Camera's up vector (y-axis), ensure it is orthogonal to the other two axes
    up = np.cross(forward, right)
    
    # Create the rotation matrix by combining the camera axes to form a basis
    rotation = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Create the translation matrix
    translation = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])

    # The view matrix is the inverse of the camera's transformation matrix
    # Here we assume the rotation matrix is orthogonal (i.e., rotation.T == rotation^-1)
    view_matrix = rotation.T @ translation

    return view_matrix

def generate_camera_poses(num_poses, angle_step, major_radius, minor_radius, camera_motion_mode, inverse=False):
    """
    Generate a camera pose that rotates around the origin, forming an elliptical trajectory. 
    """
    poses = []
    camera_positions = []

    for i in range(num_poses):
        angle = np.deg2rad(angle_step * i if not inverse else 360 - angle_step * i)

        if camera_motion_mode == 'horizontal':
            cam_x = major_radius* np.sin(angle)
            cam_y = 0
            cam_z =  minor_radius* np.cos(angle)
        elif camera_motion_mode == 'vertical':
            cam_y = major_radius * np.sin(angle)
            cam_x = 0
            cam_z =  minor_radius * np.cos(angle)
        elif camera_motion_mode == 'zoomin':
            cam_x = 0
            cam_y = 0
            cam_z =  minor_radius * np.cos(angle)
        elif camera_motion_mode == 'zoomout':
            cam_x = 0
            cam_y = 0
            cam_z =  minor_radius * (1+np.sin(angle))
        else:
            raise NotImplementedError
        
        look_at = np.array([0, 0, 0])  
        camera_position = np.array([cam_x, cam_y, cam_z])
        up_direction = np.array([0, 1, 0])  
        
        pose_matrix = look_at_matrix(camera_position, look_at, up_direction)
        poses.append(pose_matrix)
    return poses

def save_warped_image(save_path,images_lists,depth_lists,num_frames, degrees_per_frame,major_radius, minor_radius,camera_motion_mode):

    num_frames = 25
    poses = generate_camera_poses(num_frames, degrees_per_frame,major_radius, minor_radius,camera_motion_mode)

    near=0.0001
    far=10000.
    focal = 260.
    K = np.eye(3)
    K[0,0] = focal; K[1,1] = focal; K[0,2] = 1024./2; K[1,2] = 576./2

    image_o = PIL.Image.open(images_lists[0])

    pose_s = poses[0]
    cond_image = []
    masks = []
    flow12_list = []
    depth_list = []

    trans_coordinates_list = []
    trans_valid_list = []

    for i, pose_t in enumerate(poses[1:]):
        image = PIL.Image.open(images_lists[i+1])
        image = np.array(image)
        depth = np.load(depth_lists[i+1]).astype(np.float32)
        depth[depth < 1e-5] = 1e-5
        depth = 10000./depth 
        depth = np.clip(depth, near, far)
        depth_list.append(depth.copy())
        warped_frame2, mask2,flow12, trans_coordinates, trans_valid = forward_warp(image, None, depth, pose_s, pose_t, K, None)

        trans_coordinates_list.append(trans_coordinates)
        trans_valid_list.append(trans_valid)

        # flow12_list.append(flow12.copy())
        # np.save(os.path.join(save_path,str(i).zfill(4)+"_flow.npy"), flow12)

        # mask = 1-mask2
        # mask[mask < 0.5] = 0
        # mask[mask >= 0.5] = 1
        # mask = np.repeat(mask[:,:,np.newaxis]*255.,repeats=3,axis=2)
    
        # kernel = np.ones((5,5), np.uint8)
        # mask_erosion = cv2.dilate(np.array(mask), kernel, iterations = 1)
        # mask_erosion = PIL.Image.fromarray(np.uint8(mask_erosion))
        # mask_erosion.save(os.path.join(save_path,str(i).zfill(4)+"_mask.png"))

        # mask_erosion_ = np.array(mask_erosion)/255.
        # mask_erosion_[mask_erosion_ < 0.5] = 0
        # mask_erosion_[mask_erosion_ >= 0.5] = 1
        # warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2))
        # warped_frame2 = PIL.Image.fromarray(np.uint8(warped_frame2*(1-mask_erosion_)))
        # warped_frame2.save(os.path.join(save_path,str(i).zfill(4)+".png"))

        # cond_image.append(warped_frame2.copy())

        # mask_erosion = np.mean(mask_erosion_,axis = -1)
        # mask_erosion = mask_erosion.reshape(72,8,128,8).transpose(0,2,1,3).reshape(72,128,64)
        # mask_erosion = np.mean(mask_erosion,axis=2)
        # mask_erosion[mask_erosion < 0.2] = 0
        # mask_erosion[mask_erosion >= 0.2] = 1
        # masks.append(torch.from_numpy(mask_erosion).unsqueeze(0))

    

    # masks = torch.cat(masks)

    first_frame = create_grid(576, 1024)
    trans_coordinates_list = [first_frame] + trans_coordinates_list
    trans_coordinates = np.stack(trans_coordinates_list, axis=0)

    first_frame_valid = trans_valid_list[0]+True
    trans_valid_list = [first_frame_valid] + trans_valid_list
    trans_valid = np.stack(trans_valid_list, axis=0)

    np.save(os.path.join(save_path, 'trans_coordinates.npy'), trans_coordinates)
    np.save(os.path.join(save_path, 'trans_valid.npy'), trans_valid)

    return image_o,masks,cond_image

def cameractrl(image_path, depth_path, save_path, num_frames, degrees_per_frame=-0.5, major_radius=200, minor_radius=200, camera_motion_mode='horizontal'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images_lists = [image_path] * num_frames
    depth_lists = [depth_path] * num_frames

    save_warped_image(save_path,images_lists,depth_lists,num_frames,
                                            degrees_per_frame,major_radius, minor_radius,camera_motion_mode)
    print(f"Trajectory extraction finished, saved to {save_path}")
