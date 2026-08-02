import torch
from torch import Tensor
from jaxtyping import Float, Bool
import torch.nn.functional as F


def get_smooth_loss(disp, img, no_mean=False):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    ref: https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L202
    """
    if no_mean:
        out = torch.zeros_like(disp)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if no_mean:
        out[:, :, :, :-1] = out[:, :, :, :-1] + grad_disp_x
        out[:, :, :-1, :] = out[:, :, :-1, :] + grad_disp_y

        return out

    return grad_disp_x.mean() + grad_disp_y.mean()

def get_image_edge_weight_map(
        img: Float[Tensor, "*#batch 3 height width"],
        alpha: float = 1.
) -> Float[Tensor, "*#batch height width"]:
    # [B, 3, H, W-1]->[B, H, W-1] and [B, 3, H-1, W]->[B, H-1, W]
    grad_img_x = torch.mean(torch.abs(img[..., :, :-1] - img[..., :, 1:]), -3)
    grad_img_y = torch.mean(torch.abs(img[..., :-1, :] - img[..., 1:, :]),-3)

    grad_img_x = torch.exp(-alpha * grad_img_x)
    grad_img_y = torch.exp(-alpha * grad_img_y)

    # replicate instead of setting to all 1s or 0s to avoid big/small weights
    last_col = grad_img_x[..., -1:]
    last_row = grad_img_y[..., -1:, :]

    grad_img_x = torch.cat([grad_img_x, last_col], dim=-1)
    grad_img_y = torch.cat([grad_img_y, last_row], dim=-2)

    return grad_img_x + grad_img_y

def get_smooth_loss_from_depth(
        depth: Float[Tensor, "batch 1 height width"],
        img: Float[Tensor, "batch 3 height width"],
        nonorm=False
):
    mask_x, mask_y = get_valid_disp_grad_masks(depth)
    
    disp = torch.zeros_like(depth)
    valid_mask = depth > 0
    disp[valid_mask] = 1. / depth[valid_mask]

    if not nonorm:
        sum_disp = disp.sum(2, True).sum(3, True)
        valid_pixels = valid_mask.float().sum(2, True).sum(3, True)
        mean_disp = sum_disp / (valid_pixels + 1e-7)
        disp = disp / (mean_disp + 1e-7)

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    grad_disp_x = grad_disp_x * mask_x
    grad_disp_y = grad_disp_y * mask_y

    mean_x = grad_disp_x.sum() / (mask_x.sum() + 1e-7)
    mean_y = grad_disp_y.sum() / (mask_y.sum() + 1e-7)
    
    return mean_x + mean_y


def get_valid_disp_grad_masks(
        depth: Float[Tensor, "batch 1 height width"],
) -> tuple[Bool[Tensor, "batch 1 height width-1"],
           Bool[Tensor, "batch 1 height-1 width"]]:
    valid_depths = depth > 0
    
    mask_x = valid_depths[:, :, :, :-1] & valid_depths[:, :, :, 1:]
    mask_y = valid_depths[:, :, :-1, :] & valid_depths[:, :, 1:, :]
    
    return mask_x, mask_y