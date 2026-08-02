import numpy as np
from ..geometry.projection import depth_to_normal
from jaxtyping import Float
from torch import Tensor
import torch

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_rend_depth_errors(
        gt: Float[Tensor, "batch view height width"],
        rend: Float[Tensor, "batch view height width"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        height: int, width: int
):
    """Computation of error metrics between rendered and ground truth depths 
    [valid].detach().cpu().numpy()
    """

    depth_gt = gt[0]  # [V, H, W]
    rend_depths = rend[0]  # [V, H, W]

    near = near[..., None, None][0]  # [V, 1, 1]
    far = far[..., None, None][0]  # [V, 1, 1]

    valid = (depth_gt >= near) & (depth_gt <= far) & (rend_depths > 0)

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(depth_gt[valid].detach().cpu().numpy(),
                                                                       rend_depths[valid].detach().cpu().numpy())

    # [B, V, H-2, W-2, 3]
    gt_depth_normal_map, gt_depth_normal_map_mask = depth_to_normal(gt, intrinsics, height, width)
    rend_depth_normal_map, rend_depth_normal_map_mask = depth_to_normal(rend, intrinsics, height, width)
    mask = gt_depth_normal_map_mask * rend_depth_normal_map_mask

    # [B, V, H-2, W-2, 3]-># [B, V, H-2, W-2]
    dot_prod_map = (gt_depth_normal_map * rend_depth_normal_map).sum(dim=-1)
    dot_prod_map = dot_prod_map[mask].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.rad2deg(torch.acos(dot_prod_map))

    mae = angles.mean().item()
    median_ae = angles.median().item()

    rend_normal_completeness = rend_depth_normal_map_mask.float().mean().item()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, mae, median_ae, rend_normal_completeness
