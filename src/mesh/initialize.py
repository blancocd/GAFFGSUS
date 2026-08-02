# adapted from GOF and GGGS
import torch
from typing import Literal
from ..model.types import Gaussians, Vertices
from jaxtyping import Float
from torch import Tensor
from einops import einsum
from ..model.decoder.gggs_cuda_splatting import evaluate_transmittance_cuda
from ..model.decoder.gof_cuda_splatting import integrate_cuda
from ..model.encoder.common.gaussians import quaternion_to_matrix_wxyz
from . import Meshes
from functools import partial
import gdel3d
from tetranerf.utils.extension import cpp
from .tetmesh import marching_tetrahedra

@torch.no_grad()
def get_frustum_mask(points: torch.Tensor, intrinsics, view_matrices, image_shape, near: float = 0.1, far: float = 1e6):
    H, W = image_shape

    ones = torch.ones_like(points[:, 0]).unsqueeze(-1)
    # homo_points: (N, 4)
    homo_points = torch.cat([points, ones], dim=-1)

    # uv_points: (n_view, N, 4, 4)
    # Apply batch matrix multiplication to get uv_points for all cameras
    view_points = einsum(view_matrices, homo_points, "n_view b c, N c -> n_view N b")
    view_points = view_points[:, :, :3]

    uv_points = einsum(intrinsics, view_points, "n_view b c, n_view N c -> n_view N b")

    z = uv_points[:, :, -1:]
    uv_points = uv_points[:, :, :2] / z
    u, v = uv_points[:, :, 0], uv_points[:, :, 1]

    # Optionally, we can apply near-far culling
    # Apply near-far culling
    depth = view_points[:, :, -1]
    cull_near_fars = (depth >= near) & (depth <= far)

    # Apply frustum mask
    mask = torch.any(cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1), dim=0)
    return mask

def get_tetra_points(means, scales, rots, opacities, extrinsics, intrinsics, image_shape, opacity_threshold=0.0, corner_type: Literal['box', 'tetrahedron', 'octahedron', 'center_only'] = 'tetrahedron'):
    H, W = image_shape

    if opacity_threshold > 0.0:
        opac_squeeze = opacities.squeeze(-1) # [G]
        mask = opac_squeeze >= opacity_threshold
        
        valid_gaussian_indices = torch.nonzero(mask).squeeze(-1)
        rots_mat = quaternion_to_matrix_wxyz(rots[mask]) # [~G, 3, 3]
        xyz = means[mask]                                # [~G, 3]
        scale = scales[mask] * 3.0                       # [~G, 3]
    else:
        num_gaussians = means.shape[0]
        valid_gaussian_indices = torch.arange(num_gaussians, device=means.device)
        rots_mat = quaternion_to_matrix_wxyz(rots)       # [G, 3, 3]
        xyz = means                                      # [G, 3]
        scale = scales * 3.0                             # [G, 3]
        
    num_gaussians = rots_mat.shape[0]

    if corner_type == 'box':
        vertices = torch.tensor([[-1., -1., -1.], [-1., -1.,  1.], [-1.,  1., -1.], [-1.,  1.,  1.],
                                 [ 1., -1., -1.], [ 1., -1.,  1.], [ 1.,  1., -1.], [ 1.,  1.,  1.]], 
                                 device=means.device, dtype=torch.float32).T   # [3, 8]
    elif corner_type == 'tetrahedron':
        vertices = torch.tensor([[ 1.,  1.,  1.], [ 1., -1., -1.], [-1.,  1., -1.], [-1., -1.,  1.]],
                                 device=means.device, dtype=torch.float32).T   # [3, 4]
    elif corner_type == 'octahedron':
        vertices = torch.tensor([[ 1.,  0.,  0.], [-1.,  0.,  0.], [ 0.,  1.,  0.], [ 0., -1.,  0.],
                                 [ 0.,  0.,  1.], [ 0.,  0., -1.]],
                                 device=means.device, dtype=torch.float32).T   # [3, 6]
    elif corner_type == 'center_only':
        vertices = torch.empty((3, 0), device=means.device, dtype=torch.float32) # [3, 0]

    num_vertices_around = vertices.shape[-1]
    
    if num_vertices_around > 0:
        vertices = vertices.unsqueeze(0).repeat(num_gaussians, 1, 1) # G, 3, V
        vertices = vertices * scale.unsqueeze(-1) # [G,3,V]
        vertices = torch.bmm(rots_mat, vertices).squeeze(-1) + xyz.unsqueeze(-1) #[G,3,V]
        vertices = vertices.permute(0, 2, 1).reshape(-1, 3).contiguous() #[G*V,3]
        # concat center points
        vertices = torch.cat([vertices, xyz], dim=0) #[G*(V+1),3]
    else:
        vertices = xyz #[G,3]
    
    corner_g_indices = valid_gaussian_indices.repeat_interleave(num_vertices_around)
    center_g_indices = valid_gaussian_indices
    point_to_gaussian = torch.cat([corner_g_indices, center_g_indices], dim=0) # [G*(V+1)]
    
    scale_max = scale.max(dim=-1, keepdim=True)[0] #[G,3]->[G,1]
    scale_corner = scale_max.repeat(1, num_vertices_around).reshape(-1, 1) #[G,V]->[G*V,1]
    vertices_scale = torch.cat([scale_corner, scale_max], dim=0) #[G*(V+1), 1]

    view_matrix = extrinsics.inverse()
    tmp_intr = intrinsics.clone() 
    tmp_intr[:, 0, :] = tmp_intr[:, 0, :] * W
    tmp_intr[:, 1, :] = tmp_intr[:, 1, :] * H
    vertex_mask = get_frustum_mask(vertices.detach(), tmp_intr, view_matrix, image_shape)
    
    return vertices[vertex_mask], vertices_scale[vertex_mask], point_to_gaussian[vertex_mask]

@torch.no_grad()
def evaluate_alpha_gggs(points, means, scales, rots, harmonics, opacities, extrinsics, intrinsics, near, far, image_shape):

    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device=points.device)
    
    # alphas has shape [B*V, G] where B is assumed to be 1
    alphas, inside_masks = evaluate_transmittance_cuda(points, extrinsics, intrinsics, near, far, image_shape, 
                                                       means, None, scales, rots, harmonics, opacities)
    final_alpha = torch.min(alphas, dim=0)[0] #[G]
    return 0.5 - final_alpha

@torch.no_grad()
def evaluage_alpha_gof(points, means, scales, rots, harmonics, opacities, extrinsics, intrinsics, near, far, image_shape):

    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device=points.device)
    
    # alphas has shape [B*V, G] where B is assumed to be 1
    render_pkgs, alphas, colors = integrate_cuda(points, extrinsics, intrinsics, near, far, image_shape, 
                                                 means, None, scales, rots, harmonics, opacities)
    final_alpha = torch.min(alphas, dim=0)[0] #[G]
    return 0.5 - final_alpha


@torch.no_grad()
def compute_initial_mesh_sdf_with_binary_search(
    points, points_scale, cells, sdf_function, 
    num_binary_steps=10, filter_large_edges=True
):
    sdf = sdf_function(points=points)
    vertices = points.unsqueeze(0)
    tets = cells.long()
    sdf_batched = sdf.unsqueeze(0)
    
    scale_batched = points_scale.view(1, -1, 1)

    verts_list, scale_list, faces_list, interp_v_list = marching_tetrahedra(
        vertices, tets, sdf_batched, scale_batched
    )
    
    end_points, end_sdf = verts_list[0]
    end_scales = scale_list[0]
    faces = faces_list[0]
    interp_v = interp_v_list[0]
    
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    
    left_scale = end_scales[:, 0, 0]
    right_scale = end_scales[:, 1, 0]
    
    distance = torch.norm(left_points - right_points, dim=-1)
    scale = left_scale + right_scale

    for step in range(num_binary_steps):
        mid_points = (left_points + right_points) * 0.5
        mid_sdf = sdf_function(points=mid_points).unsqueeze(-1)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))
        
        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        
        ind_low_flat = ind_low.flatten()
        left_points[ind_low_flat] = mid_points[ind_low_flat]
        right_points[~ind_low_flat] = mid_points[~ind_low_flat]

    refined_verts = (left_points + right_points) * 0.5

    if filter_large_edges:
        valid_vertex_mask = distance <= scale
        face_mask = valid_vertex_mask[faces].all(dim=1)
        faces = faces[face_mask]

    return refined_verts, faces, interp_v


@torch.inference_mode()
def initialize_mesh(
    gaussians: Gaussians, 
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    near: Float[Tensor, " batch view"],
    far: Float[Tensor, " batch view"],
    image_shape: tuple[int, int], # h, w
    decoder: Literal['gggs', 'gof'] = 'gggs',
    opacity_threshold: float=0.00,
    num_binary_steps: int=10,
    filter_large_edges=True,
    use_gpu_del=False
):
    if decoder == 'gggs':
        print("Using GGGS mesh extraction!")
        sdf_function = partial(
            evaluate_alpha_gggs,
            gaussians=gaussians,
            extrinsics=extrinsics[0], 
            intrinsics=intrinsics[0], 
            near=near[0], 
            far=far[0], 
            image_shape=image_shape
        )
    else:
        print("Using GOF mesh extraction!")
        sdf_function = partial(
            evaluage_alpha_gof,
            gaussians=gaussians,
            extrinsics=extrinsics[0], 
            intrinsics=intrinsics[0], 
            near=near[0], 
            far=far[0], 
            image_shape=image_shape
        )

    points, points_scale, point_to_gaussian = get_tetra_points(
        gaussians, extrinsics[0], intrinsics[0], 
        image_shape, opacity_threshold=opacity_threshold
    )

    points = points.contiguous()
    if use_gpu_del:
        points = torch.unique(points, dim=0)
        try:
            tets = gdel3d.triangulate_device(points)
        except RuntimeError as e:
            print(f"Skipping degenerate batch: {e}")
        valid_mask = (cells < points.shape[0]).all(dim=1)
        cells = cells[valid_mask].long()
    else:
        cells = cpp.triangulate(points)

    refined_verts, faces, interp_v = compute_initial_mesh_sdf_with_binary_search(
        points=points, 
        points_scale=points_scale, 
        cells=cells, 
        sdf_function=sdf_function, 
        num_binary_steps=num_binary_steps, 
        filter_large_edges=filter_large_edges
    )

    mesh_vertex_gaussian_indices = point_to_gaussian[interp_v]

    return Meshes(verts=refined_verts, faces=faces), mesh_vertex_gaussian_indices


def initialize_mesh_vertices(
    gaussians: Gaussians, 
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    near: Float[Tensor, "batch view"],
    far: Float[Tensor, "batch view"],
    image_shape: tuple[int, int], # h, w
    decoder: Literal['gggs', 'gof'] = 'gggs',
    opacity_threshold: float=0.00,
    num_binary_steps: int=10,
    filter_large_edges=True,
    use_gpu_del=False,
    remove_duplicates=False,
    vertex_corner_type: Literal['box', 'tetrahedron', 'octahedron', 'center_only'] = 'tetrahedron'
) -> Vertices:
    
        
    B = gaussians.means.shape[0]

    # batch gaussian 3 d_sh -> batch gaussian d_sh 3
    global_harmonics = gaussians.harmonics.transpose(-2, -1).contiguous()
    
    batch_verts = []
    batch_sh = []
    batch_op = []
    batch_faces = []
    # batch_mappings = []
    
    for b in range(B):
        means = gaussians.means[b]
        scales = gaussians.scales[b]
        rots = gaussians.rotations_world[b]
        harmonics = gaussians.harmonics[b]
        opacities = gaussians.opacities[b] # shape [G]
        
        ext_b = extrinsics[b]
        int_b = intrinsics[b]
        near_b = near[b]
        far_b = far[b]

        if decoder == 'gggs':
            sdf_function = partial(evaluate_alpha_gggs, means=means, scales=scales, rots=rots, harmonics=harmonics, 
                                    opacities=opacities, extrinsics=ext_b, intrinsics=int_b, near=near_b, far=far_b, image_shape=image_shape)
        else:
            sdf_function = partial(evaluage_alpha_gof, means=means, scales=scales, rots=rots, harmonics=harmonics, 
                                    opacities=opacities, extrinsics=ext_b, intrinsics=int_b, near=near_b, far=far_b, image_shape=image_shape)

        points, points_scale, point_to_gaussian = get_tetra_points(
            means, scales, rots, opacities, ext_b, int_b, image_shape, opacity_threshold, corner_type=vertex_corner_type
        )

        if points.shape[0] < 4:
            batch_verts.append(torch.empty((0, 3), dtype=torch.float32, device=points.device))
            d_sh = global_harmonics.shape[2] 
            batch_sh.append(torch.empty((0, d_sh, 3), dtype=torch.float32, device=points.device))
            batch_op.append(torch.empty((0,), dtype=torch.float32, device=points.device))
            batch_faces.append(torch.empty((0, 3), dtype=torch.long, device=points.device))
            continue
        
        if remove_duplicates:
            with torch.no_grad():
                _, inverse_indices = torch.unique(points.detach(), dim=0, return_inverse=True)
                
                num_unique = inverse_indices.max() + 1
                
                unique_to_orig_idx = torch.empty(num_unique, dtype=torch.long, device=points.device)
                unique_to_orig_idx.scatter_(0, inverse_indices, torch.arange(points.shape[0], device=points.device))
                
            points = points[unique_to_orig_idx]
            points_scale = points_scale[unique_to_orig_idx]
            point_to_gaussian = point_to_gaussian[unique_to_orig_idx]

        points = points.contiguous()
        points_scale = points_scale.contiguous()
        point_to_gaussian = point_to_gaussian.contiguous()

        with torch.no_grad():
            if use_gpu_del:
                cells = gdel3d.triangulate_host(points)
                valid_mask = (cells < points.shape[0]).all(dim=1)
                cells = cells[valid_mask].long()
            else:
                cells = cpp.triangulate(points)

            refined_verts_nograd, faces, interp_v = compute_initial_mesh_sdf_with_binary_search(
                points=points, points_scale=points_scale, cells=cells, 
                sdf_function=sdf_function, num_binary_steps=num_binary_steps, filter_large_edges=filter_large_edges
            )

        left_indices = interp_v[:, 0]
        right_indices = interp_v[:, 1]
        
        with torch.no_grad():
            sdf_diff = sdf_function(points=points)
            sdf_left = sdf_diff[left_indices]
            sdf_right = sdf_diff[right_indices]
            w_right = sdf_left / (sdf_left - sdf_right + 1e-8)
            w_left = 1.0 - w_right
        
        left_pts_diff = points[left_indices]
        right_pts_diff = points[right_indices]
        
        v_linear = left_pts_diff * w_left.unsqueeze(-1) + right_pts_diff * w_right.unsqueeze(-1)
        refined_verts = v_linear + (refined_verts_nograd - v_linear).detach()

        diff_total = right_pts_diff - left_pts_diff
        total_dist = torch.sqrt(torch.sum(diff_total**2, dim=-1, keepdim=True) + 1e-8)
        diff = refined_verts - left_pts_diff
        dist_from_left = torch.sqrt(torch.sum(diff**2, dim=-1, keepdim=True) + 1e-8)
        alpha = (dist_from_left / total_dist).clamp(0.0, 1.0)
        
        g_left_idx = point_to_gaussian[left_indices]
        g_right_idx = point_to_gaussian[right_indices]
        
        left_sh, right_sh = global_harmonics[b][g_left_idx], global_harmonics[b][g_right_idx]
        left_op, right_op = opacities[g_left_idx], opacities[g_right_idx]
        
        alpha_sh = alpha.view(-1, 1, 1) 
        mesh_sh = (1.0 - alpha_sh) * left_sh + alpha_sh * right_sh
        alpha_op = alpha.view(-1)
        mesh_op = (1.0 - alpha_op) * left_op + alpha_op * right_op

        batch_verts.append(refined_verts)
        batch_sh.append(mesh_sh)
        batch_op.append(mesh_op)
        batch_faces.append(faces)

    vertices = Vertices(
        positions=batch_verts,    
        harmonics=batch_sh,       
        weights=batch_op,         
        faces=batch_faces,        
        scale_mults=None                         
    )

    return vertices


# Define the SH constant for degree 0
SH_C0 = 0.28209479177387814

def convert_vertices_to_meshes(vertices, batch_idx: int = 0) -> Meshes:
    """
    Extracts a single scene from the Vertices dataclass and converts it 
    into a Meshes object ready for Nvdiffrast rendering.
    """
    # Extract geometry
    verts = vertices.positions[batch_idx]
    faces = vertices.faces[batch_idx]
    
    # If the mesh is empty, return an empty Meshes object
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        return Meshes(
            verts=torch.empty((0, 3), device=verts.device),
            faces=torch.empty((0, 3), dtype=torch.int32, device=faces.device)
        )

    # Extract color from Spherical Harmonics (DC component is at index 0)
    # harmonics shape: [V, d_sh, 3] -> DC shape: [V, 3]
    sh_dc = vertices.harmonics[batch_idx][:, 0, :]
    
    # Convert SH degree 0 to RGB and clamp to valid [0, 1] range
    verts_colors = (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

    # Instantiate your Meshes class
    return Meshes(verts=verts, faces=faces, verts_colors=verts_colors)