from math import isqrt
from typing import Literal

import torch
from diff_triangle_rasterization import TriangleRasterizationSettings, TriangleRasterizer
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ...geometry.projection import get_fov, homogenize_points

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    vertices_positions: list[Tensor],
    triangles_indices: list[Tensor],
    vertices_sh_coefficients: list[Tensor],
    vertices_weights: list[Tensor],
    scale_invariant: bool = True,
    use_sh: bool = True,
    sigma: float = 1.,
    alpha_normalization_threshold: float | None = None
):
    assert len(vertices_sh_coefficients) > 0
    assert use_sh or vertices_sh_coefficients[0].shape[-1] == 1

    # Get SH degree from the first tensor in the list
    # Shape of items is [Vertices, 3, d_sh]
    _, n, _ = vertices_sh_coefficients[0].shape
    degree = isqrt(n) - 1
    shs = vertices_sh_coefficients

    b, _, _ = extrinsics.shape
    h_init, w_init = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    # NEED to upsample, otherwise triangles fall through the pixels and are not rendered
    upsample = 14
    h = upsample * h_init
    w = upsample * w_init

    all_images = []
    all_rendered_normals = []
    all_expected_depths = []
    all_median_depths = []
    all_alphas = []
    for i in range(b):
        # per-triangle scale buffer returned from the rasterizer
        scaling = torch.zeros_like(triangles_indices[i][:, 0], dtype=torch.float32, requires_grad=True).detach()
        settings = TriangleRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = TriangleRasterizer(settings)

        # Indexing [i] works perfectly on lists to grab the [Vertices, Dim] tensor!
        rendered_image, radii, scaling, allmap, max_blending, was_rendered = rasterizer(
            vertices=vertices_positions[i],
            triangles_indices=triangles_indices[i].to(torch.int32),
            vertex_weights=vertices_weights[i],
            sigma=sigma,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i][:, 0, :], # Access 0th SH degree
            scaling=scaling 
        )

        render_alpha = allmap[1:2] # 1, H, W
        if alpha_normalization_threshold is not None:
            normalized_img = rendered_image / render_alpha.clamp(min=1e-5)
            normalized_img = torch.clamp(normalized_img, 0.0, 1.0)
            rendered_image = torch.where(
                render_alpha > alpha_normalization_threshold,
                normalized_img,
                rendered_image
            )

        img_hr = rendered_image.unsqueeze(0)  # -> [1, 3, H, W]
        img_ds_area = F.interpolate(img_hr, size=(h_init, w_init), mode="area")  # [1, 3, H0, W0]
        rendered_image = img_ds_area.squeeze(0)

        alpha_hr = render_alpha.unsqueeze(0) # -> [1, 1, H, W]
        alpha_ds_area = F.interpolate(alpha_hr, size=(h_init, w_init), mode="area")
        render_alpha = alpha_ds_area.squeeze(0) # -> [1, H0, W0]

        render_alpha = render_alpha.squeeze(0) # [h0,w0]
        render_alpha = torch.nan_to_num(render_alpha, 0, 0)

        render_normal = allmap[2:5] # 3, H, W
        img_hr = render_normal.unsqueeze(0)  # -> [1, 3, H, W]
        img_ds_area = F.interpolate(img_hr, size=(h_init, w_init), mode="area")  # [1, 3, H0, W0]
        render_normal = img_ds_area.squeeze(0)

        render_expected_depth = allmap[0:1] # 1, H, W
        img_hr = render_expected_depth.unsqueeze(0)  # -> [1, 1, H, W]
        img_ds_area = F.interpolate(img_hr, size=(h_init, w_init), mode="area")  # [1, 1, H0, W0]
        render_expected_depth = img_ds_area.squeeze(0).squeeze(0)
        render_expected_depth = torch.nan_to_num(render_expected_depth, 0, 0)

        render_depth_median = allmap[5:6] # 1, H, W
        img_hr = render_depth_median.unsqueeze(0)  # -> [1, 1, H, W]
        img_ds_area = F.interpolate(img_hr, size=(h_init, w_init), mode="area")  # [1, 1, H0, W0]
        render_depth_median = img_ds_area.squeeze(0).squeeze(0)
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        all_images.append(rendered_image)
        all_rendered_normals.append(render_normal)
        all_expected_depths.append(render_expected_depth)
        all_median_depths.append(render_depth_median)
        all_alphas.append(render_alpha)
    return torch.stack(all_images), torch.stack(all_rendered_normals), torch.stack(all_expected_depths), torch.stack(all_median_depths), torch.stack(all_alphas)

