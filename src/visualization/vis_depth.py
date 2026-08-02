import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# https://github.com/autonomousvision/unimatch/blob/master/utils/visualization.py


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def viz_depth_tensor(disp, return_numpy=False, colormap="plasma", return_cb_fig=False):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()

    # now needed since viz mesh depths and
    # nvdiffrast returns 0 depth for regions without proj mesh
    finite_mask = np.isfinite(disp)
    valid_pixels = disp[finite_mask]
    
    # Return black image if no valid pixels
    if valid_pixels.size == 0:
        black_image = np.zeros((*disp.shape, 3), dtype=np.uint8)
        if return_numpy:
            return black_image
        viz = torch.from_numpy(black_image).permute(2, 0, 1)  # [3, H, W]
        return viz
    
    vmax = np.percentile(valid_pixels, 95)
    vmin = valid_pixels.min()
        
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(
        np.uint8
    )  # [H, W, 3]

    out_im = colormapped_im if return_numpy else torch.from_numpy(colormapped_im).permute(2, 0, 1)

    if return_cb_fig:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(mapper, cax=ax, orientation='horizontal')
        cb.set_label('MAE (Pred, GT)')
        return out_im, fig

    return out_im

def viz_diff_tensor(diff, return_numpy=False, colormap="RdBu_r", return_cb_fig=False):
    assert isinstance(diff, torch.Tensor)
    diff = diff.numpy()

    finite_mask = np.isfinite(diff)
    valid_pixels = diff[finite_mask]
    
    vmax = max(np.percentile(valid_pixels, 95), 1e-5)
    vmin = min(np.percentile(valid_pixels, 5), -1e-5)
    
    normalizer = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    
    colormapped_im = (mapper.to_rgba(diff)[:, :, :3] * 255).astype(np.uint8)

    out_im = colormapped_im if return_numpy else torch.from_numpy(colormapped_im).permute(2, 0, 1)

    if return_cb_fig:
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        cb = fig.colorbar(mapper, cax=ax, orientation='horizontal')
        cb.set_label('Depth Difference (Pred - GT)')
        return out_im, fig

    return out_im