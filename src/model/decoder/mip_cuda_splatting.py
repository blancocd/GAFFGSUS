from math import isqrt
from typing import Literal

import torch
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except:
    pass
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor

from ...geometry.projection import get_fov, get_projection_matrix

def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"] | None,
    gaussian_scaling: Float[Tensor, "batch gaussian 3"] | None,
    gaussian_rotation: Float[Tensor, "batch gaussian 4"] | None,
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = False,
    use_sh: bool = True,
    return_radii: bool = False
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    assert scale_invariant is False

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = None if gaussian_covariances is None else \
            gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_scaling = None if gaussian_scaling is None else \
            gaussian_scaling * scale[:, None, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    subpixel_offset = torch.zeros((h, w, 2), dtype=torch.float32, device=extrinsics.device)

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i])
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            kernel_size=0.1,
            subpixel_offset=subpixel_offset,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        cov3D_precomp = None if gaussian_covariances is None else gaussian_covariances[i, :, row, col]
        rotations = None if gaussian_rotation is None else gaussian_rotation[i]
        scales = None if gaussian_scaling is None else gaussian_scaling[i]

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=cov3D_precomp,
            scales=scales,
            rotations=rotations
        )
        all_images.append(image)
        all_radii.append(radii)

    if return_radii:
        return torch.stack(all_images), torch.stack(all_radii)
    return torch.stack(all_images)
