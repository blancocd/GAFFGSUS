import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def back_project(depth, intrinsics):
    # Back project 2D pixel coords to 3D points
    # depth: [B, H, W]
    # intrinsics: [B, 3, 3]
    b, h, w = depth.shape
    grid = coords_grid(b, h, w, homogeneous=True, device=depth.device)  # [B, 3, H, W]

    intrinsics_inv = torch.inverse(intrinsics)  # [B, 3, 3]

    points = intrinsics_inv.bmm(grid.view(b, 3, -1)).view(b, 3, h, w) * depth.unsqueeze(1)  # [B, 3, H, W]

    return points


def camera_transform(points_ref, extrinsics_ref=None, extrinsics_tgt=None, extrinsics_rel=None):
    # Transform 3D points from reference camera to target camera
    # points_ref: [B, 3, H, W]
    # extrinsics_ref: [B, 4, 4]
    # extrinsics_tgt: [B, 4, 4]
    # extrinsics_rel: [B, 4, 4], relative pose transform
    b, _, h, w = points_ref.shape

    if extrinsics_rel is None:
        extrinsics_rel = torch.bmm(extrinsics_tgt, torch.inverse(extrinsics_ref))  # [B, 4, 4]

    points_tgt = torch.bmm(extrinsics_rel[:, :3, :3],
                           points_ref.view(b, 3, -1)) + extrinsics_rel[:, :3, -1:]  # [B, 3, H*W]

    points_tgt = points_tgt.view(b, 3, h, w)  # [B, 3, H, W]

    return points_tgt


def reproject(points_tgt, intrinsics, return_mask=False, return_depth=False):
    # reproject to target view
    # points_tgt: [B, 3, H, W]
    # intrinsics: [B, 3, 3]

    b, _, h, w = points_tgt.shape

    proj_points = torch.bmm(intrinsics, points_tgt.view(b, 3, -1)).view(b, 3, h, w)  # [B, 3, H, W]

    X = proj_points[:, 0]
    Y = proj_points[:, 1]
    depth = proj_points[:, 2] # [B, H, W]
    valid_z = depth > 1e-3
    Z = depth.clamp(min=1e-3)

    pixel_coords = torch.stack([X / Z, Y / Z], dim=1).view(b, 2, h, w)  # [B, 2, H, W] in image scale

    out = [pixel_coords]

    if return_mask:
        # valid mask in pixel space
        mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] <= (w - 1)) & (
                pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] <= (h - 1)) & valid_z # [B, H, W]
        out.append(mask)    

    if return_depth:
        out.append(depth)
    
    return tuple(out) if len(out) > 1 else out[0]


def reproject_coords(depth_ref, intrinsics, extrinsics_ref=None, extrinsics_tgt=None, extrinsics_rel=None,
                     return_mask=False, depth_tgt=None):
    # Compute reprojection sample coords
    points_ref = back_project(depth_ref, intrinsics)  # [B, 3, H, W]
    points_tgt = camera_transform(points_ref, extrinsics_ref, extrinsics_tgt, extrinsics_rel=extrinsics_rel)

    if return_mask:
        if depth_tgt is not None:
              # [B, 2, H, W], [B, H, W], [B, H, W]
            reproj_coords, mask, depth_ref_in_tgt = reproject(points_tgt, intrinsics,
                                                              return_mask=return_mask,
                                                              return_depth=True)
            # [B, H, W]
            depth_tgt_ref_order = bilinear_sample(depth_tgt.unsqueeze(1), reproj_coords, 
                                                  mode='nearest').squeeze(1)
            
            # tgt depth will always be closer or at same depth, threshold is there as to not
            # cull too many points which are actually at the same depth but whose depth
            # appers a tiny bit closer in the tgt depth map
            threshold = 0.07 * depth_ref_in_tgt
            visible_mask = depth_tgt_ref_order - depth_ref_in_tgt + threshold > 0

            mask *= visible_mask
        else:
            reproj_coords, mask = reproject(points_tgt, intrinsics,
                                            return_mask=return_mask)  # [B, 2, H, W] in image scale
        
        return reproj_coords, mask

    reproj_coords = reproject(points_tgt, intrinsics,
                              return_mask=return_mask)  # [B, 2, H, W] in image scale

    return reproj_coords


def warp_with_depth_pose(feature1, intrinsics, pose, depth,
                         padding_mode='zeros',
                         return_rigid_flow=False,
                         return_mask=False,
                         depth_tgt=None
                         ):
    assert depth.dim() == 3  # [B, H, W]
    sample_coords = reproject_coords(depth,
                                     intrinsics,
                                     extrinsics_rel=pose,
                                     return_mask=return_mask,
                                     depth_tgt=depth_tgt
                                     )  # [B, 2, H, W]

    if return_mask:
        sample_coords, mask = sample_coords

    sample_coords = sample_coords.permute(0, 2, 3, 1)  # [B, H, W, 2]

    warped_feature1 = bilinear_sample(feature1, sample_coords,
                                      padding_mode=padding_mode)  # [B, C, H, W]

    if return_mask:
        return warped_feature1, mask

    if return_rigid_flow:
        b, h, w = depth.size()
        coords_init = coords_grid(b, h, w, device=depth.device)  # [B, 2, H, W]
        rigid_flow = sample_coords.permute(0, 3, 1, 2) - coords_init

        return warped_feature1, rigid_flow

    return warped_feature1
