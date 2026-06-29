#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from tribench.vendor.triangle_splatting.scene.cameras import Camera
import numpy as np
import torch
from tribench.vendor.triangle_splatting.utils.general_utils import PILtoTorch
from tribench.vendor.triangle_splatting.utils.graphics_utils import fov2focal
from tribench.vendor.training_images import is_dtu_scene, load_rgba_for_training

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    channels = cam_info.image.split()
    is_dtu = is_dtu_scene(args)
    dtu_eval_mode = str(getattr(args, "dtu_eval_mode", "full")).lower()
    use_alpha = (not is_dtu) or dtu_eval_mode == "foreground"
    if len(channels) > 3 and use_alpha:
        resized_image_rgb, loaded_mask = load_rgba_for_training(
            cam_info.image,
            resolution,
            PILtoTorch,
            composite_white=is_dtu,
        )
        gt_image = resized_image_rgb
    else:
        if len(channels) >= 3:
            resized_image_rgb = torch.cat(
                [PILtoTorch(channel, resolution) for channel in channels[:3]],
                dim=0,
            )
        else:
            resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    if use_alpha and loaded_mask is None and getattr(cam_info, "mask", None) is not None:
        mask_arr = cam_info.mask
        if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
            mask_arr = mask_arr[..., 0]
        from PIL import Image as _PILImage
        mask_pil = _PILImage.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
        if mask_pil.size != resolution:
            mask_pil = mask_pil.resize(resolution, _PILImage.NEAREST)
        loaded_mask = torch.from_numpy(
            np.asarray(mask_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
