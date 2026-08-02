from typing import List, Optional
import torch
import nvdiffrast.torch as dr
from jaxtyping import Float
from torch import Tensor
from ..geometry.projection import get_fov, get_projection_matrix
from einops import rearrange
from tqdm import tqdm

def transform_points_world_to_view(
    points:torch.Tensor,
    extrinsics: Float[Tensor, "batch 4 4"],
    use_p3d_convention:bool=False,
):
    """Transform points from world space to view space.

    Args:
        points (torch.Tensor): Should have shape (n_cameras, N, 3).
        cameras (List[Camera]): List of Cameras. Should contain n_cameras elements.
        use_p3d_convention (bool, optional): Defaults to False.
        
    Returns:
        torch.Tensor: Has shape (n_cameras, N, 3).
    """
    world_view_transforms = rearrange(extrinsics.inverse(), "b i j -> b j i")
    
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (n_cameras, N, 4)
    view_points = (points_h @ world_view_transforms)[..., :3]  # (n_cameras, N, 3)
    if use_p3d_convention:
        factors = torch.tensor([[[-1, -1, 1]]], device=points.device)  # (1, 1, 3)
        view_points = factors * view_points  # (n_cameras, N, 3)
    return view_points

# single image
def nvdiff_rasterization(
    camera_mtx:Float[Tensor, "4 4"],
    image_height:int, 
    image_width:int,
    verts:torch.Tensor, 
    faces:torch.Tensor,
    return_indices_only:bool=False,
    glctx=None,
    return_rast_out:bool=False,
    return_positions:bool=False,
):
    device = verts.device
    
    # Convert to homogeneous coordinates
    pos = torch.cat([verts, torch.ones([verts.shape[0], 1], device=device)], axis=1)

    # Transform points to NDC/clip space
    pos = torch.matmul(pos, camera_mtx)[None]
    
    # Rasterize with NVDiffRast
    # TODO: WARNING: pix_to_face is not in the correct range [-1, F-1] but in [0, F],
    # With 0 indicating that no triangle was hit.
    # So we need to subtract 1.
    rast_out, _ = dr.rasterize(glctx, pos=pos, tri=faces, resolution=[image_height, image_width])
    bary_coords, zbuf, pix_to_face = rast_out[..., :2], rast_out[..., 2], rast_out[..., 3].int() - 1
    
    if return_indices_only:
        return pix_to_face
    
    _output = (bary_coords, zbuf, pix_to_face)
    if return_rast_out:
        _output = _output + (rast_out,)
    if return_positions:
        _output = _output + (pos,)
    return _output


class Meshes(torch.nn.Module):
    """
    Meshes class for storing meshes parameters.
    """
    def __init__(
        self, 
        verts:torch.Tensor, 
        faces:torch.Tensor, 
        verts_colors:torch.Tensor|None=None
    ):
        super().__init__()
        assert verts_colors is None or verts_colors.shape[0] == verts.shape[0]
        self.verts = verts
        self.faces = faces.to(torch.int32)
        self.verts_colors = verts_colors
        
    @property
    def face_normals(self):
        faces_verts = self.verts[self.faces]
        faces_verts_normals = torch.cross(
            faces_verts[:,1] - faces_verts[:,0], 
            faces_verts[:,2] - faces_verts[:,0], 
            dim=-1
        )
        faces_verts_normals = torch.nn.functional.normalize(faces_verts_normals, dim=-1)
        return faces_verts_normals
    
    @property
    def vertex_normals(self):
        raise NotImplementedError("Vertex normals are not implemented yet")
    
    def submesh(
        self, 
        vert_idx:Optional[torch.Tensor]=None, 
        face_idx:Optional[torch.Tensor]=None,
        vert_mask:Optional[torch.Tensor]=None,
        face_mask:Optional[torch.Tensor]=None,
    ):
        assert (
            (vert_idx is not None) or (vert_mask is not None) 
            or
            (face_idx is not None) or (face_mask is not None)
        ), "Either vert_idx, vert_mask, face_idx, or face_mask must be provided"

        if (vert_idx is not None) or (vert_mask is not None):
            if vert_mask is None:
                vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
                vert_mask[vert_idx] = True
            face_mask = vert_mask[self.faces].all(dim=1)

        elif (face_idx is not None) or (face_mask is not None):
            if face_mask is None:
                face_mask = torch.zeros(self.faces.shape[0], dtype=torch.bool, device=self.verts.device)
                face_mask[face_idx] = True
            vert_mask = torch.zeros(self.verts.shape[0], dtype=torch.bool, device=self.verts.device)
            vert_mask[self.faces[face_mask]] = True
        
        old_vert_idx_to_new_vert_idx = torch.zeros(self.verts.shape[0], dtype=self.faces.dtype, device=self.verts.device)
        old_vert_idx_to_new_vert_idx[vert_mask] = torch.arange(vert_mask.sum(), dtype=self.faces.dtype, device=self.verts.device)
        
        new_verts = self.verts[vert_mask]
        new_verts_colors = None if self.verts_colors is None else self.verts_colors[vert_mask]
        new_faces = old_vert_idx_to_new_vert_idx[self.faces][face_mask]
        
        return Meshes(verts=new_verts, faces=new_faces, verts_colors=new_verts_colors)


def combine_meshes(
    meshes:List[Meshes],
) -> Meshes:
    """Combines multiple meshes into a single mesh.

    Args:
        meshes (List[Meshes]): List of meshes to combine.

    Returns:
        Meshes: Combined mesh.
    """
    all_verts = torch.zeros(0, 3, dtype=meshes[0].verts.dtype, device=meshes[0].verts.device)
    all_faces = torch.zeros(0, 3, dtype=meshes[0].faces.dtype, device=meshes[0].faces.device)
    all_verts_colors = None if meshes[0].verts_colors is None else torch.zeros(0, 3, dtype=meshes[0].verts_colors.dtype, device=meshes[0].verts_colors.device)
    
    n_total_verts = 0
    
    for _, mesh in enumerate(meshes):
        all_verts = torch.cat([all_verts, mesh.verts], dim=0)
        all_faces = torch.cat([all_faces, mesh.faces + n_total_verts], dim=0)
        if all_verts_colors is not None:
            all_verts_colors = torch.cat([all_verts_colors, mesh.verts_colors], dim=0)
        n_total_verts += mesh.verts.shape[0]
    
    return Meshes(verts=all_verts, faces=all_faces, verts_colors=all_verts_colors)


class RasterizationSettings():
    """
    Rasterization settings for meshes.
    """
    def __init__(
        self, 
        image_size=(1080, 1920),
        blur_radius=0.0,
        faces_per_pixel=1,
        ):
        self.image_size = image_size
        self.blur_radius = blur_radius
        self.faces_per_pixel = faces_per_pixel


class Fragments():
    def __init__(self, bary_coords, zbuf, pix_to_face):
        self.bary_coords = bary_coords  # Shape (1, height, width, 1, 3)
        self.zbuf = zbuf  # Shape (1, height, width, 1)
        self.pix_to_face = pix_to_face  # Shape (1, height, width, 1)


class MeshRasterizer(torch.nn.Module):
    """
    Class for rasterizing meshes with NVDiffRast.
    """
    def __init__(
        self, 
        raster_settings:RasterizationSettings,
        use_opengl=True,
    ):
        super().__init__()
        
        if raster_settings is None:
            raster_settings = RasterizationSettings()
        self.raster_settings = raster_settings
        self.height, self.width = raster_settings.image_size
        self.cameras = None
        
        if use_opengl:
            self.gl_context = dr.RasterizeGLContext()
        else:
            self.gl_context = dr.RasterizeCudaContext()
            
    def forward(
        self, 
        mesh:Meshes, 
        camera_mtx:Float[Tensor, "4 4"],
        cam_idx:int=0,
        return_only_pix_to_face:bool=False,
        return_rast_out:bool=False,
        return_positions:bool=False,
    ):
        height, width = self.height, self.width
        nvdiff_rast_out = nvdiff_rasterization(
            camera_mtx=camera_mtx,
            image_height=height, 
            image_width=width,
            verts=mesh.verts,
            faces=mesh.faces,
            return_indices_only=False,
            glctx=self.gl_context,
            return_rast_out=return_rast_out,
            return_positions=return_positions,
        )
        bary_coords, zbuf, pix_to_face = nvdiff_rast_out[:3]
        if return_rast_out:
            rast_out = nvdiff_rast_out[3]
        if return_positions:
            pos = nvdiff_rast_out[4]
        
        if return_only_pix_to_face:
            return pix_to_face.view(1, height, width, 1)
        bary_coords = torch.cat([bary_coords, 1. - bary_coords.sum(dim=-1, keepdim=True)], dim=-1)
        
        # TODO: Zbuf is still in NDC space, should convert to camera space
        fragments = Fragments(
            bary_coords.view(1, height, width, 1, 3),
            zbuf.view(1, height, width, 1),
            pix_to_face.view(1, height, width, 1),
        )
        _output = (fragments,)
        if return_rast_out:
            _output = _output + (rast_out,)
        if return_positions:
            _output = _output + (pos,)
        return _output
        
    
class MeshRenderer(torch.nn.Module):
    """
    Class for rendering meshes with NVDiffRast and a shader.
    """
    def __init__(self, rasterizer:MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh:Meshes, 
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, " batch"],
        far: Float[Tensor, " batch"],
        cam_idx=0,
        return_depth=False,
        return_colors=False,
        return_normals=False,
        return_white_texture=False,
        use_antialiasing=True,
        return_pix_to_face=False,
        check_errors=True,
    ):
        b, _, _ = extrinsics.shape
        
        output_pkg = {}
        if return_depth:
            output_pkg["depth"] = []
            verts_depth = transform_points_world_to_view(mesh.verts, extrinsics)[..., 2]
        if return_colors:
            output_pkg["rgb"] = []
        if return_normals:
            output_pkg["normals"] = []
        if return_pix_to_face:
            output_pkg["pix_to_face"] = []
        if return_white_texture:
            output_pkg["white_texture"] = []

        # Get full projection matrices
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        camera_mtx = view_matrix @ projection_matrix

        for i in range(b):
            fragments, rast_out, pos = self.rasterizer(mesh, camera_mtx[i], cam_idx, return_rast_out=True, return_positions=True)
            
            # return_colors = mesh.verts_colors is not None

            # Compute per-vertex features to render
            features = torch.zeros(mesh.verts.shape[0], 0, device=mesh.verts.device)        
            
            if return_depth:
                depth_idx = features.shape[-1]
                curr_verts_depth = verts_depth[i]  # Shape (N, )
                features = torch.cat([features, curr_verts_depth.view(mesh.verts.shape[0], 1)], dim=-1)
                
            if return_colors:
                color_idx = features.shape[-1]
                features = torch.cat([features, mesh.verts_colors], dim=-1)  # Shape (N, n_features)

            if return_white_texture:
                white_idx = features.shape[-1]
                verts_white = torch.ones(mesh.verts.shape[0], 3, device=mesh.verts.device)
                features = torch.cat([features, verts_white], dim=-1)
            
            # Compute image
            feature_img, _ = dr.interpolate(features[None], rast_out, mesh.faces)  # Shape (1, H, W, n_features)
            
            # Antialiasing for propagating gradients
            if use_antialiasing:
                feature_img = dr.antialias(feature_img, rast_out, pos, mesh.faces)  # Shape (1, H, W, n_features)
            
            if return_depth:
                output_pkg["depth"].append(feature_img[..., depth_idx:depth_idx+1])  # Shape (1, H, W)
            if return_colors:
                output_pkg["rgb"].append(feature_img[..., color_idx:color_idx+3])  # Shape (1, H, W, 3)
                
            if return_white_texture:
                output_pkg["white_texture"].append(feature_img[..., white_idx:white_idx+3]) # Shape (1, H, W, 3)

            # Compute per-face normals
            if return_normals:
                valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
                if check_errors:
                    error_mask = fragments.pix_to_face >= mesh.faces.shape[0]
                    error_encountered = torch.sum(error_mask)
                    if error_encountered > 0:
                        print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                        fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=mesh.faces.shape[0] - 1)
                        valid_mask = valid_mask & ~error_mask
                output_pkg["normals"].append(mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask)  # Shape (1, H, W, 3)
                # if use_antialiasing:
                #     output_pkg["normals"] = dr.antialias(output_pkg["normals"], rast_out, pos, mesh.faces)  # Shape (1, H, W, 3)
                
            if return_pix_to_face:
                output_pkg["pix_to_face"].append(fragments.pix_to_face)

        return output_pkg


    @torch.no_grad()
    def get_nonsky_faces(
        self,
        mesh: Meshes,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        camera_indices: torch.Tensor,
        mask_dir: str,
        erosion_kernel_size: int = 5,
        votes: int = 2,
    ) -> Tensor:
        """
        Returns mask tensor of mesh faces which dont project 'sky' regions of the provided 2D masks.
        Faces are labeled as sky faces if the project to sky regions at least 'votes' times
        """
        import os
        from PIL import Image
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        
        b, _, _ = extrinsics.shape
        device = mesh.verts.device
        image_shape = (self.rasterizer.height, self.rasterizer.width)
        
        # 1. Initialize a boolean mask for faces mapped to the sky (default to False)
        is_sky_face = torch.zeros(mesh.faces.shape[0], dtype=torch.int32, device=device)
        
        # 2. Get sorted list of mask files
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.JPG'))])
        if not mask_files:
            raise ValueError(f"No mask images found in {mask_dir}")
            
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        camera_mtx = view_matrix @ projection_matrix
        
        for i in range(b):
            # Rasterize to get the face ID buffer
            pix_to_face = self.rasterizer(
                mesh, 
                camera_mtx[i], 
                return_only_pix_to_face=True
            ) # (1, height, width, 1)
            pix_to_face = pix_to_face.squeeze() # Shape: [height, width]
            
            # -- Load and process the mask --
            idx = camera_indices[i].item()
            mask_path = os.path.join(mask_dir, mask_files[idx])
            print(mask_path)
            
            # Load mask: 0 is sky, >0 is nonsky
            mask_img = Image.open(mask_path).convert('L')
            mask_tensor = TF.to_tensor(mask_img).unsqueeze(0).to(device) # Shape: [1, 1, H_orig, W_orig]
            mask_tensor = (mask_tensor >= 0.5).float()
            
            # Erode the sky area (dilate the nonsky area) to protect foreground edges
            if erosion_kernel_size > 1:
                padding = erosion_kernel_size // 2
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size=erosion_kernel_size, stride=1, padding=padding)

            # Resize mask to match rasterized image shape
            mask_tensor = F.interpolate(mask_tensor, size=image_shape, mode='nearest')
                
            # -- Find faces in the sky --
            # Sky is where the processed mask is exactly 0
            sky_pixels = (mask_tensor.squeeze() == 0)
            
            # Get face IDs at those sky pixels
            sky_face_ids = pix_to_face[sky_pixels]
            
            # Filter out the background (-1) and any out-of-bounds indices
            valid_mask = (sky_face_ids >= 0) & (sky_face_ids < mesh.faces.shape[0])
            sky_face_ids = sky_face_ids[valid_mask].long()
            unique_sky_ids = torch.unique(sky_face_ids)
            
            # Mark these faces for deletion
            is_sky_face[unique_sky_ids] += 1

        # Only cull faces which were seen as sky in at least two views.
        face_mask = ~(is_sky_face >= votes)

        print(f"Identified {(~face_mask).sum().item()}/{mesh.faces.shape[0]} faces as sky artifacts.")
        
        return face_mask

def fuse_fragments(fragments1:Fragments, fragments2:Fragments):
    raster_mask1 = fragments1.pix_to_face > -1
    raster_mask2 = fragments2.pix_to_face > -1
    
    # raster_mask1 = fragments1.zbuf > 0.
    # raster_mask2 = fragments2.zbuf > 0.
    
    no_raster_mask = (~raster_mask1) & (~raster_mask2)
    
    zbuf1 = torch.where(raster_mask1, fragments1.zbuf, 1000.)
    zbuf2 = torch.where(raster_mask2, fragments2.zbuf, 1000.)
    
    all_zbufs = torch.cat([zbuf1[..., None], zbuf2[..., None]], dim=-1)  # Shape (1, H, W, 1, 2)
    zbuf, argzbuf = torch.min(all_zbufs, dim=-1)  # argzbuf is of shape (1, H, W, 1)
    zbuf[no_raster_mask] = 0.
    
    all_pix_to_face = torch.cat(
        [
            fragments1.pix_to_face[..., None], 
            fragments2.pix_to_face[..., None]
        ], 
        dim=-1
    )  # Shape (1, H, W, 1, 2)
    pix_to_face = torch.gather(
        all_pix_to_face, 
        dim=-1, 
        index=argzbuf[..., None]
    )[..., 0]  # Shape (1, H, W, 1)
    pix_to_face[no_raster_mask] = -1
    
    all_bary_coords = torch.cat([fragments1.bary_coords[..., None], fragments2.bary_coords[..., None]], dim=-1)
    bary_coords = torch.gather(
        all_bary_coords, 
        dim=-1, 
        index=argzbuf[..., None, None].expand(-1, -1, -1, -1, 3, -1)
    )[..., 0]  # Shape (1, H, W, 1, 3)

    return Fragments(
        bary_coords,
        zbuf,
        pix_to_face,
    )

    
class ScalableMeshRenderer(torch.nn.Module):
    """
    Class for rendering big meshes with NVDiffRast.
    """
    def __init__(self, rasterizer: MeshRasterizer):
        super().__init__()
        self.rasterizer = rasterizer
        # TODO: Add shader
        
    def forward(
        self, 
        mesh: Meshes, 
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        cam_idx: int = 0,
        return_depth: bool = False,
        return_colors: bool = False,
        return_normals: bool = False,
        return_white_texture=False,
        use_antialiasing: bool = True,
        return_pix_to_face: bool = False,
        check_errors: bool = True,
        max_triangles_in_batch: int = 2**24  # Corresponds to the nb of triangles above which Nvdiffrast breaks
    ):
        b, _, _ = extrinsics.shape
        
        output_pkg = {}
        if return_depth:
            output_pkg["depth"] = []
            verts_depth = transform_points_world_to_view(mesh.verts, extrinsics)[..., 2]
        if return_colors:
            output_pkg["rgb"] = []
        if return_normals:
            output_pkg["normals"] = []
        if return_pix_to_face:
            output_pkg["pix_to_face"] = []

        # Get full projection matrices
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        camera_mtx = view_matrix @ projection_matrix

        for i in range(b):
            n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch
            
            fragments = None
            idx_shift = 0
            
            for i_pass in range(n_passes):
                start_idx = i_pass * max_triangles_in_batch
                end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])
                
                # Compute submesh
                sub_faces = mesh.faces[start_idx:end_idx]
                sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)
                
                # Rasterize submesh
                _fragments, _, pos = self.rasterizer(sub_mesh, camera_mtx[i], cam_idx, return_rast_out=True, return_positions=True)
                
                # Combine fragments
                if fragments is None:
                    fragments = _fragments
                else:
                    _fragments.pix_to_face = torch.where(
                        _fragments.pix_to_face > -1, 
                        _fragments.pix_to_face + idx_shift, 
                        _fragments.pix_to_face
                    )
                    fragments = fuse_fragments(fragments, _fragments)
                
                # Update idx shift
                idx_shift = idx_shift + len(sub_faces)
            
            # Filter mesh and fragments to keep only rasterized faces. 
            # Decreases the number of faces to at most H * W.
            filtered_face_idx, filtered_pix_to_face = fragments.pix_to_face.unique(return_inverse=True)
            
            # SAFETY CHECK: unique() sorts the elements. If -1 (background) is present, it will be at index 0.
            # We only slice off the first element if it is actually the background.
            if filtered_face_idx[0] == -1:
                filtered_face_idx = filtered_face_idx[1:]
                filtered_pix_to_face = filtered_pix_to_face - 1
                
            filtered_faces = mesh.faces[filtered_face_idx]
            filtered_mesh = Meshes(verts=mesh.verts, faces=filtered_faces, verts_colors=mesh.verts_colors)
            fragments.pix_to_face = filtered_pix_to_face
            
            # Rebuild rast_out
            rast_out = torch.zeros(*fragments.zbuf.shape[:-1], 4, device=fragments.zbuf.device)
            rast_out[..., :2] = fragments.bary_coords[..., 0, :2]
            rast_out[..., 2:3] = fragments.zbuf
            rast_out[..., 3:4] = fragments.pix_to_face.float() + 1
            
            return_colors_local = filtered_mesh.verts_colors is not None

            # Compute per-vertex features to render
            features = torch.zeros(filtered_mesh.verts.shape[0], 0, device=filtered_mesh.verts.device)
            
            if return_depth:
                depth_idx = features.shape[-1]
                curr_verts_depth = verts_depth[i]  # Shape (N, )
                features = torch.cat([features, curr_verts_depth.view(filtered_mesh.verts.shape[0], 1)], dim=-1)
                
            if return_colors_local:
                color_idx = features.shape[-1]
                features = torch.cat([features, filtered_mesh.verts_colors], dim=-1)  # Shape (N, n_features)

            if return_white_texture:
                white_idx = features.shape[-1]
                verts_white = torch.ones(mesh.verts.shape[0], 3, device=mesh.verts.device)
                features = torch.cat([features, verts_white], dim=-1)
            
            # Compute image
            feature_img, _ = dr.interpolate(features[None], rast_out, filtered_mesh.faces)  # Shape (1, H, W, n_features)
            
            # Antialiasing for propagating gradients
            if use_antialiasing:
                feature_img = dr.antialias(feature_img, rast_out, pos, filtered_mesh.faces)  # Shape (1, H, W, n_features)
            
            if return_depth:
                output_pkg["depth"].append(feature_img[..., depth_idx:depth_idx+1])  # Shape (1, H, W, 1)

            if return_colors_local:
                output_pkg["rgb"].append(feature_img[..., color_idx:color_idx+3])  # Shape (1, H, W, 3)

            if return_white_texture:
                output_pkg["white_texture"] = feature_img[..., white_idx:white_idx+3]  # Shape (1, H, W, 3)
                
            # Compute per-face normals
            if return_normals:
                valid_mask = fragments.pix_to_face >= 0  # Shape (1, H, W, 1)
                if check_errors:
                    error_mask = fragments.pix_to_face >= filtered_mesh.faces.shape[0]
                    error_encountered = torch.sum(error_mask)
                    if error_encountered > 0:
                        print(f"[WARNING] Rasterized {error_encountered} pixels with invalid triangle index.")
                        fragments.pix_to_face = torch.clamp(fragments.pix_to_face, min=0, max=filtered_mesh.faces.shape[0] - 1)
                        valid_mask = valid_mask & ~error_mask
                output_pkg["normals"].append(filtered_mesh.face_normals[fragments.pix_to_face].squeeze()[None] * valid_mask)  # Shape (1, H, W, 3)
                
            if return_pix_to_face:
                output_pkg["pix_to_face"].append(fragments.pix_to_face)

        return output_pkg


    def _get_fused_pix_to_face(
        self, 
        mesh: Meshes, 
        camera_mtx: Float[Tensor, "4 4"], 
        max_triangles_in_batch: int = 2**24
    ):
        """Helper to safely rasterize huge meshes in chunks and get the fused pix_to_face."""
        n_passes = (mesh.faces.shape[0] + max_triangles_in_batch - 1) // max_triangles_in_batch
        fragments = None
        idx_shift = 0
        
        print(f"Scalable mesh renderer will need {n_passes} passes.")
        for i_pass in range(n_passes):
            start_idx = i_pass * max_triangles_in_batch
            end_idx = min(start_idx + max_triangles_in_batch, mesh.faces.shape[0])
            
            sub_faces = mesh.faces[start_idx:end_idx]
            sub_mesh = Meshes(verts=mesh.verts, faces=sub_faces)
            
            # Rasterize chunk
            _fragments = self.rasterizer(sub_mesh, camera_mtx, return_rast_out=False, return_positions=False)[0]
            
            # Offset the face indices by the chunk's starting position
            _fragments.pix_to_face = torch.where(
                _fragments.pix_to_face > -1, 
                _fragments.pix_to_face + idx_shift, 
                _fragments.pix_to_face
            )
            
            # Combine fragments
            if fragments is None:
                fragments = _fragments
            else:
                fragments = fuse_fragments(fragments, _fragments)
                
            idx_shift += len(sub_faces)
            
        return fragments.pix_to_face

    @torch.no_grad()
    def get_stretched_faces_mask(
        self,
        mesh: Meshes,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        threshold: float = 0.05
    ) -> Tensor:
        """
        Returns a mask tensor to keep faces whose depth variance is below the threshold.
        Evaluates geometrically without NVDiffRast.
        """
        b, _, _ = extrinsics.shape
        device = mesh.verts.device
        num_faces = mesh.faces.shape[0]

        # Initialize minimum relative depth difference to infinity
        min_rel_diff = torch.full((num_faces,), float('inf'), dtype=torch.float32, device=device)

        # Get full projection matrices
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        camera_mtx = view_matrix @ projection_matrix

        for i in range(b):
            # 1. Transform vertices to view space (for true Z depth)
            verts_h = torch.cat([mesh.verts, torch.ones_like(mesh.verts[:, :1])], dim=-1)
            view_verts = verts_h @ view_matrix[i]
            z_view = view_verts[:, 2]

            # 2. Transform to NDC (Clip space) for geometric frustum culling
            clip_verts = verts_h @ camera_mtx[i]
            w = clip_verts[:, 3:4]
            ndc_verts = clip_verts[:, :3] / (w + 1e-8)

            # 3. Check if vertices are in frustum (NDC bounds [-1, 1] and in front of camera)
            in_frustum_v = (
                (ndc_verts[:, 0] >= -1.0) & (ndc_verts[:, 0] <= 1.0) &
                (ndc_verts[:, 1] >= -1.0) & (ndc_verts[:, 1] <= 1.0) &
                (ndc_verts[:, 2] >= -1.0) & (ndc_verts[:, 2] <= 1.0) &
                (w.squeeze() > 0)
            )

            # 4. Gather face attributes
            face_z = z_view[mesh.faces.long()]         # Shape: [F, 3]
            face_in_frustum = in_frustum_v[mesh.faces.long()] # Shape: [F, 3]

            # A face is considered valid for this camera if ANY of its vertices are in the frustum
            # AND ALL of its vertices are strictly in front of the camera (z > 0) to avoid division by zero/negatives.
            any_in_frustum = face_in_frustum.any(dim=1)
            all_positive_z = face_z.min(dim=1)[0] > 0
            valid_faces = any_in_frustum & all_positive_z

            valid_idx = torch.nonzero(valid_faces).squeeze()
            if valid_idx.numel() == 0:
                continue
            
            # 5. Compute depth variance metrics on valid faces
            valid_face_z = face_z[valid_idx] # Shape: [K, 3]
            z_max = valid_face_z.max(dim=1)[0]
            z_min = valid_face_z.min(dim=1)[0]

            # Calculate relative difference: (max - min) / min
            rel_diff = (z_max - z_min) / (z_min + 1e-6)

            # 6. Update the global minimum observed relative difference
            current_min = min_rel_diff[valid_idx]
            min_rel_diff[valid_idx] = torch.min(current_min, rel_diff)

        # 7. Final mask generation
        # Faces with min_rel_diff > threshold are heavily stretched and should be culled.
        # Faces that were NEVER in any camera's frustum remain `inf`. We KEEP them here 
        # because the 'get_seen_faces_mask' already handles unseen face culling.
        keep_mask = (min_rel_diff <= threshold) | torch.isinf(min_rel_diff)

        print(f"Depth Variance Filter: Culled {(~keep_mask).sum().item()}/{num_faces} elongated faces.")

        return keep_mask

    @torch.no_grad()
    def get_nonsky_faces(
        self,
        mesh: Meshes,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, "batch"],
        far: Float[Tensor, "batch"],
        camera_indices: torch.Tensor,
        mask_dir: str,
        erosion_kernel_size: int = 5,
        votes: int = 2,
        max_triangles_in_batch: int = 2**24
    ) -> Tensor:
        """
        Returns mask tensor of mesh faces which dont project to 'sky' regions,
        processing in chunks to handle highly dense meshes.
        """
        import os
        from PIL import Image
        import torchvision.transforms.functional as TF
        import torch.nn.functional as F
        
        b, _, _ = extrinsics.shape
        device = mesh.verts.device
        image_shape = (self.rasterizer.height, self.rasterizer.width)
        
        # 1. Initialize a counter for faces mapped to the sky
        is_sky_face = torch.zeros(mesh.faces.shape[0], dtype=torch.int32, device=device)
        
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.JPG'))])
        if not mask_files:
            raise ValueError(f"No mask images found in {mask_dir}")
            
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
        projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
        view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
        camera_mtx = view_matrix @ projection_matrix
        
        for i in tqdm(range(b)):
            # Use the chunked helper to avoid OOM on huge meshes
            pix_to_face = self._get_fused_pix_to_face(
                mesh, 
                camera_mtx[i], 
                max_triangles_in_batch
            )
            pix_to_face = pix_to_face.squeeze() # Shape: [height, width]
            
            # -- Load and process the mask --
            idx = camera_indices[i].item()
            mask_path = os.path.join(mask_dir, mask_files[idx])
            print(mask_path)
            
            mask_img = Image.open(mask_path).convert('L')
            mask_tensor = TF.to_tensor(mask_img).unsqueeze(0).to(device) 
            
            # Resize mask to match rasterized image shape
            mask_tensor = F.interpolate(mask_tensor, size=image_shape, mode='nearest')
            
            # Erode the sky area
            if erosion_kernel_size > 1:
                padding = erosion_kernel_size // 2
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size=erosion_kernel_size, stride=1, padding=padding)
                
            # -- Find faces in the sky --
            sky_pixels = (mask_tensor.squeeze() == 0)
            sky_face_ids = pix_to_face[sky_pixels]
            
            valid_mask = (sky_face_ids >= 0) & (sky_face_ids < mesh.faces.shape[0])
            sky_face_ids = sky_face_ids[valid_mask].long()
            
            # Count each face at most once per view to match the logic
            unique_sky_faces = torch.unique(sky_face_ids)
            is_sky_face[unique_sky_faces] += 1

        # Keep faces which were seen as sky less than 'votes' times
        face_mask = ~(is_sky_face >= votes)

        print(f"Identified {(~face_mask).sum().item()}/{mesh.faces.shape[0]} faces as sky artifacts.")
        
        return face_mask