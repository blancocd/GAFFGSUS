from math import isqrt

import torch
from gggs_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange
from jaxtyping import Float
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
    gaussian_scaling: Float[Tensor, "batch gaussian 3"],
    gaussian_rotation: Float[Tensor, "batch gaussian 4"] | None,
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = False,
    use_sh: bool = True,
    return_radii: bool = False,
    require_coord = True,
    require_depth = True,
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    assert scale_invariant is False # from MipSplatting impl

    # Make sure everything is in a range where numerical issues don't appear.
    # The near plane (smallest values) would be 1
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = None if gaussian_covariances is None else \
            gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_scaling = gaussian_scaling * scale[:, None, None]
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

    # Transpose rearrangements since GOF multiplies from the left to right instead of right to left
    # see forward.cu at line 194
    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    # view matrix transforms from world to camera coordinates
    # projection matrix from camera coordiantes to NDC
    # NDC coordinates maps all valid points to -1, 1 on the x and y axes and 0 to 1 on the z axis
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    # transforms from world coordinates to NDC coordinates
    full_projection = view_matrix @ projection_matrix

    # rendered_image, radii, rendered_expected_coord, rendered_median_coord, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal
    max_sg_degree = 0
    all_rendered_images, all_radii, all_rendered_median_depths, all_rendered_alphas, all_rendered_normals = [], [], [], [], []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i])
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            kernel_size=0.1,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            sg_degree=max_sg_degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            require_depth = require_depth,
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        # 3d filter on scaling and opacities would be called here, also not included in mipsplatting
        # row, col = torch.triu_indices(3, 3)

        # GOF needs the scaling and rotation as it's used in the view2gaussian transformation
        # need to include model.encoder.rotate_quat_to_world=true as otherwise the rotations wont
        # correspond to world frame but camera frame. GOF rasterizer expects everything in world coordinate frames
        # that flag also sets the order to wxyz which is also what GOF, Mip, and gsplat expect. 
        # resplat uses xyzw otherwise as well as scipy
        # using scale and rotation has better times for encoder (mostly) and decoder, confusingly

        row, col = torch.triu_indices(3, 3)
        cov3D_precomp = None if gaussian_covariances is None else gaussian_covariances[i, :, row, col]
        rotations = None if gaussian_rotation is None else gaussian_rotation[i]

        num_pts = gaussian_means.shape[1]
        sg_axis = torch.zeros(num_pts, max_sg_degree, 3).float().cuda()
        sg_axis = torch.nn.functional.normalize(sg_axis, dim=2)
        sg_sharpness = torch.zeros(num_pts, max_sg_degree).float().cuda()
        sg_color = torch.zeros(num_pts, max_sg_degree, 3).float().cuda()

        rendered_image, radii, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            sg_axis = sg_axis,
            sg_sharpness = sg_sharpness,
            sg_color = sg_color,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            scales=gaussian_scaling[i],
            rotations=rotations,
        )
        all_rendered_images.append(rendered_image)
        all_radii.append(radii)
        all_rendered_median_depths.append(rendered_median_depth)
        all_rendered_alphas.append(rendered_alpha)
        all_rendered_normals.append(rendered_normal)

    return torch.stack(all_rendered_images), torch.stack(all_radii), torch.stack(all_rendered_median_depths), torch.stack(all_rendered_alphas), torch.stack(all_rendered_normals)

# here because we are assuming batch=1 then there's no need to repeat the gaussian atribs
# extrinsics, intrinsics, near, and far are reshaped to be B*V followed by other dimensions
def evaluate_transmittance_cuda(
    points3D: Float[Tensor, "points 3"],
    extrinsics: Float[Tensor, "view 4 4"],
    intrinsics: Float[Tensor, "view 3 3"],
    near: Float[Tensor, " view"],
    far: Float[Tensor, " view"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "gaussian 3"],
    gaussian_covariances: Float[Tensor, "gaussian 3 3"] | None,
    gaussian_scaling: Float[Tensor, "gaussian 3"],
    gaussian_rotation: Float[Tensor, "gaussian 4"] | None,
    gaussian_sh_coefficients: Float[Tensor, "gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "gaussian"],
    background_color: Float[Tensor, "view 3"] | None = None,
    scale_invariant: bool = False,
    use_sh: bool = True,
    return_radii: bool = False
):
    if background_color is None:
        background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=extrinsics.device).unsqueeze(0).repeat(extrinsics.shape[0], 1)

    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    assert scale_invariant is False # from MipSplatting impl

    # Make sure everything is in a range where numerical issues don't appear.
    # The near plane (smallest values) would be 1
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = None if gaussian_covariances is None else \
            gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_scaling = gaussian_scaling * scale[:, None, None]
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    # Transpose rearrangements since GOF multiplies from the left to right instead of right to left
    # see forward.cu at line 194
    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    # view matrix transforms from world to camera coordinates
    # projection matrix from camera coordiantes to NDC
    # NDC coordinates maps all valid points to -1, 1 on the x and y axes and 0 to 1 on the z axis
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    # transforms from world coordinates to NDC coordinates
    full_projection = view_matrix @ projection_matrix

    all_transmittance, all_inside = [], []
    max_sg_degree = 0
    for i in range(b):
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            kernel_size=0.1,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            sg_degree=max_sg_degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False, # kernel_size!=0,  # Original usage has it as False but if we compute new scales from kernel size we are already filtering
            debug=False,
            require_depth = True,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        cov3D_precomp = None if gaussian_covariances is None else gaussian_covariances[:, row, col]
        rotations = None if gaussian_rotation is None else gaussian_rotation

        transmittance, inside = rasterizer.evaluate_transmittance(
            points3D=points3D,
            means3D=gaussian_means,
            opacities=gaussian_opacities[..., None],
            scales=gaussian_scaling,
            rotations=rotations,
            cov3Ds_precomp=cov3D_precomp,
        )

        all_transmittance.append(transmittance)
        all_inside.append(inside)

    return 1-torch.stack(all_transmittance), torch.stack(all_inside)
