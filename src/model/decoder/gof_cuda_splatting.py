from math import isqrt
from typing import Literal

import torch
from gof_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points, get_projection_matrix

# extrinsics, intrinsics, near, and far are reshaped to be B*V followed by other dimensions
# gaussian attrbs are repeated so they are B*V, G, where G=V*H*W, they repeat them because if
# doing [B, V, H, W]->[B*V, H*W] instead the rendering would be only on the gaussians that
# spawned for each view and not all views in that scene
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
    return_radii: bool = False
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

    # render time 3d dilation didn't have much effect and could interfere with marching tetrahedra
    # from learned sdf values as sdf values are learned at distances determined by the scale
    # if kernel_size > 0:
    #     # this works because there are V copies of the original [B, G] gaussians from the encoder
    #     # so each extrinsics is properly applied to the scene batch
    #     # [B*V, 4, 4], [B*V, G, 4] -> [B*V, G, 4]
    #     # extrinsics.inverse is the transformation from world to camera
    #     gaussian_means_cam = einsum(extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i")

    #     # project to screen space
    #     # [B*V, G]
    #     x, y, z = gaussian_means_cam[..., 0], gaussian_means_cam[..., 1], gaussian_means_cam[..., 2]
    #     z = torch.clamp(z, min=0.001)
        
    #     # [B*V]
    #     focal_length_x = intrinsics[..., 0, 0] * w
    #     focal_length_y = intrinsics[..., 1, 1] * h

    #     # [B*V, G] / [B*V, G] * [B*V, 1] + 1 -> [B*V, G]
    #     x = x / z * focal_length_x.unsqueeze(-1) + w / 2.0
    #     y = y / z * focal_length_y.unsqueeze(-1) + h / 2.0
        
    #     # use similar tangent space filtering as in the paper
    #     in_screen = torch.logical_and( # [B*V, G]
    #         torch.logical_and(x >= -0.15 * w, x <= w * 1.15),
    #         torch.logical_and(y >= -0.15 * h, y <= 1.15 * h))
        
    #     scales_square = torch.square(gaussian_scaling) # [B*V, G, 3]
    #     det1 = scales_square.prod(dim=-1) # [B*V, G]


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

    subpixel_offset = torch.zeros((h, w, 2), dtype=torch.float32, device=extrinsics.device)

    all_render_pkgs = []
    all_radii = []
    for i in range(b):
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            kernel_size=0.0,
            subpixel_offset=subpixel_offset,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False, # kernel_size!=0,  # Original usage has it as False but if we compute new scales from kernel size we are already filtering
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        # make sure projected gaussians are large enough
        # render time 3d dilation didn't have much effect and could interfere with marching tetrahedra
        # from learned sdf values as sdf values are learned at distances determined by the scale
        # if kernel_size > 0:
        #     curr_focal_length_x = focal_length_x[i] # [1]
        #     # [G]
        #     curr_in_screen = in_screen[i]
        #     # [B*V, G]->[G]->[mG]
        #     gaussian_means_inscreen = gaussian_means[i][curr_in_screen]
        #     mean_gradients = torch.zeros_like(gaussian_means_inscreen)
        #     shs_inscreen = shs[i][curr_in_screen]
        #     opacities_inscreen = gaussian_opacities[i][curr_in_screen]
        #     scales_inscreen = gaussian_scaling[i][curr_in_screen]
        #     rotations_inscreen = gaussian_rotation[i][curr_in_screen]

        #     depth_inscreen = z[i][curr_in_screen] # [mG]
        #     det1_inscreen = det1[i][curr_in_screen] # [mG]
        #     scales_square_inscreen = scales_square[i][curr_in_screen] # [mG]

        #     # [mG]
        #     kernel_size = torch.tensor(kernel_size, device=intrinsics.device)
        #     filter_3d = torch.sqrt(kernel_size) * depth_inscreen / curr_focal_length_x

        #     # [mG, 3] + [mG, 1] -> [mG, 3]
        #     scales_square_after_filter_3d = scales_square_inscreen + torch.square(filter_3d).unsqueeze(-1)
        #     det2 = scales_square_after_filter_3d.prod(dim=-1) # [mG]

        #     coef = torch.sqrt(det1_inscreen / det2) # [mG]
        #     opacities_inscreen = opacities_inscreen * coef # [mG]

        #     scales_inscreen = torch.sqrt(scales_square_after_filter_3d)

        #     render_pkg, radii = rasterizer(
        #         means3D=gaussian_means_inscreen,
        #         means2D=mean_gradients,
        #         shs=shs_inscreen if use_sh else None,
        #         colors_precomp=None if use_sh else shs_inscreen[:, 0, :],
        #         opacities=opacities_inscreen[..., None],
        #         cov3D_precomp=None,
        #         scales=scales_inscreen,
        #         rotations=rotations_inscreen,
        #     )

        # else:
            # GOF needs the scaling and rotation as it's used in the view2gaussian transformation
            # need to include model.encoder.rotate_quat_to_world=true as otherwise the rotations wont
            # correspond to world frame but camera frame. GOF rasterizer expects everything in world coordinate frames
            # that flag also sets the order to wxyz which is also what GOF, Mip, and gsplat expect. 
            # resplat uses xyzw otherwise as well as scipy
            # using scale and rotation has better times for encoder (mostly) and decoder, confusingly

        mean_gradients = torch.zeros_like(gaussian_means[i])
        row, col = torch.triu_indices(3, 3)
        cov3D_precomp = None if gaussian_covariances is None else gaussian_covariances[i, :, row, col]
        rotations = None if gaussian_rotation is None else gaussian_rotation[i]

        render_pkg, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=cov3D_precomp,
            scales=gaussian_scaling[i],
            rotations=rotations,
        )

        all_render_pkgs.append(render_pkg)
        all_radii.append(radii)

    if return_radii:
        return torch.stack(all_render_pkgs), torch.stack(all_radii)
    return torch.stack(all_render_pkgs)

# here because we are assuming batch=1 then there's no need to repeat the gaussian atribs
# extrinsics, intrinsics, near, and far are reshaped to be B*V followed by other dimensions
def integrate_cuda(
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
    shs = rearrange(gaussian_sh_coefficients, "g xyz n -> g n xyz").contiguous()

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

    subpixel_offset = torch.zeros((h, w, 2), dtype=torch.float32, device=extrinsics.device)

    all_render_pkgs, all_alpha, all_color, all_radii = [], [], [], []
    for i in range(b):
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            kernel_size=0.0,
            subpixel_offset=subpixel_offset,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False, # kernel_size!=0,  # Original usage has it as False but if we compute new scales from kernel size we are already filtering
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        mean_gradients = torch.zeros_like(gaussian_means)
        row, col = torch.triu_indices(3, 3)
        cov3D_precomp = None if gaussian_covariances is None else gaussian_covariances[:, row, col]
        rotations = None if gaussian_rotation is None else gaussian_rotation

        render_pkg, alpha_integrated, color_integrated, radii = rasterizer.integrate(
            points3D=points3D,
            means3D=gaussian_means,
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[:, 0, :],
            opacities=gaussian_opacities[..., None],
            cov3D_precomp=cov3D_precomp,
            scales=gaussian_scaling,
            rotations=rotations,
        )

        all_render_pkgs.append(render_pkg)
        all_alpha.append(alpha_integrated)
        all_color.append(color_integrated)
        all_radii.append(radii)

    if return_radii:
        return torch.stack(all_render_pkgs), torch.stack(all_alpha), torch.stack(all_color), torch.stack(all_radii)
    return torch.stack(all_render_pkgs), torch.stack(all_alpha), torch.stack(all_color)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)
