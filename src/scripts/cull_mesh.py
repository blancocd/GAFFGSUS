import argparse
import torch
import trimesh
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from einops import rearrange, einsum
import json

# Clean, absolute imports starting from the root 'src' package
from src.mesh import (
    Meshes, 
    RasterizationSettings, 
    MeshRasterizer, 
    ScalableMeshRenderer
)
from src.dataset.dataset_dl3dv import DatasetDL3DVCfg, DatasetDL3DV
from src.dataset.view_sampler.view_sampler_all import ViewSamplerAllCfg, ViewSamplerAll

def cull_mesh_for_scene(mesh_path: str, scene_name: str, output_path: str, dataset_roots: list[str], 
                        mask_dir: str, scale:int=1, votes:int=3, erosion_kernel_size:int=3,
                        absdiff_threshold:float=0.05, pow2_triangles:int=24, index_path='',
                        no_cull_stretch=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading mesh from {mesh_path}", flush=True)
    mesh = trimesh.load(mesh_path, process=False)
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device)

    meshes = Meshes(verts=verts, faces=faces)
    
    opencv_pose_format='dl3dv' in dataset_roots[0]
    width = 640 if 'mipnerf360' in dataset_roots[0] else 540
    view_cfg = ViewSamplerAllCfg(name='all')
    dataset_cfg = DatasetDL3DVCfg(
        name="dl3dv",
        roots=[Path(r) for r in dataset_roots],
        baseline_epsilon=1e-3,
        max_fov=100.0,
        make_baseline_1=False,
        augment=True,
        test_len=-1,
        test_chunk_interval=1,
        train_times_per_scene=1,
        test_times_per_scene=1,
        ori_image_shape=[width, 960],

        max_size=[512,960],
        min_size=[384,512],
        random_crop = False,
        near = 0.01,
        far = 200.,
        baseline_scale_bounds = False,

        image_shape=[width, 960], # Expected rasterization shape
        background_color=[0.0, 0.0, 0.0],
        cameras_are_circular=False,
        overfit_to_scene=scene_name, #load only such scene
        view_sampler=view_cfg,
        opencv_pose_format=opencv_pose_format
    )
    
    # Use ViewSamplerAll to grab every camera in the scene chunk
    view_sampler = ViewSamplerAll(view_cfg, stage='test', is_overfitting=True,
                                  cameras_are_circular=True, step_tracker=True)
    
    # Stage "test" bypasses random shuffling
    dataset = DatasetDL3DV(cfg=dataset_cfg, stage="test", view_sampler=view_sampler)
    
    # 3. Iterate dataset once to grab the scene
    print(f"Fetching camera poses for scene: {scene_name}", flush=True)
    scene_data = next(iter(dataset))
    
    # Concatenate cameras from context and target just in case the sampler splits them
    extrinsics = scene_data["context"]["extrinsics"].to(device) # v, 4, 4
    intrinsics = scene_data["context"]["intrinsics"].to(device)
    near = scene_data["context"]["near"].to(device)
    far = scene_data["context"]["far"].to(device)
    indices = scene_data["context"]["index"].to(device)
    
    if len(index_path) != 0:
        with open(index_path, 'r') as f:
            json_data = json.load(f)
        train_indices = json_data.get(scene_name, {}).get('context', [])
        test_indices = json_data.get(scene_name, {}).get('target', [])
        train_indices = train_indices+test_indices

        # select only train indices
        extrinsics = extrinsics[train_indices]
        intrinsics = intrinsics[train_indices]
        near = near[train_indices]
        far = far[train_indices]
        indices = indices[train_indices]

        print("Train indices from index path:", train_indices)
    print("Train indices from batch:", indices.cpu().numpy())
    
    print(f"Total cameras selected for projection: {extrinsics.shape[0]}", flush=True)
    
    # scale up so subpixel faces (at native resolution or lower res) are seen
    image_shape = (dataset_cfg.image_shape[0]*scale, dataset_cfg.image_shape[1]*scale)
    raster_settings = RasterizationSettings(image_size=image_shape)
    rasterizer = MeshRasterizer(raster_settings, use_opengl=False).to(device)
    renderer = ScalableMeshRenderer(rasterizer).to(device)
    
    if not no_cull_stretch:
        print("Culling highly stretched faces based on depth variance", flush=True)
        depth_variance_mask = renderer.get_stretched_faces_mask(
            mesh=meshes, 
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            near=near, 
            far=far,
            threshold=absdiff_threshold # Adjust this value if it is too aggressive or lenient
        )
        meshes = meshes.submesh(face_mask=depth_variance_mask)

    if len(mask_dir) > 0:
        print("Culling sky faces", flush=True)
        non_sky_faces_mask = renderer.get_nonsky_faces(
            mesh=meshes, 
            extrinsics=extrinsics, 
            intrinsics=intrinsics, 
            near=near, 
            far=far, 
            camera_indices=indices, 
            mask_dir=mask_dir,
            erosion_kernel_size=erosion_kernel_size,
            votes=votes,
            max_triangles_in_batch=2**pow2_triangles
        )
        # NOTE: erosion_kernel_size=5 means it shrinks the sky mask by ~2 pixels. Increase to 7 or 9 if it still cuts foreground.

    culled_mesh = meshes.submesh(face_mask=non_sky_faces_mask)
    culled_mesh = trimesh.Trimesh(vertices=culled_mesh.verts.cpu().numpy(), faces=culled_mesh.faces.cpu().numpy())
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    culled_mesh.export(output_path)
    print(f"Saving culled mesh to {output_path}", flush=True)
    print("Done!", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cull unseen and sky faces from a mesh.")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to the input mesh (.ply, .obj, etc.)")
    parser.add_argument("--scene_name", type=str, required=True, help="Name of the DL3DV scene key (e.g. hash string)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output culled mesh")
    parser.add_argument("--dataset_roots", type=str, nargs='+', required=True, help="Path(s) to the DL3DV dataset root")
    parser.add_argument("--mask_dir", type=str, default='', help="Path to the nonsky_masks_4 directory")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--erosion_kernel_size", type=int, default=3)
    parser.add_argument("--absdiff_threshold", type=float, default=0.01)
    parser.add_argument("--pow2_triangles", type=int, default=20)
    parser.add_argument("--index_path", type=str, default='')
    parser.add_argument("--no_cull_stretch", action='store_true', help="Skip culling of stretched faces based on depth variance")
    
    args = parser.parse_args()
    
    cull_mesh_for_scene(
        mesh_path=args.mesh_path,
        scene_name=args.scene_name,
        output_path=args.output_path,
        dataset_roots=args.dataset_roots,
        mask_dir=args.mask_dir,
        scale=args.scale,
        votes=args.votes,
        erosion_kernel_size=args.erosion_kernel_size,
        absdiff_threshold=args.absdiff_threshold,
        pow2_triangles=args.pow2_triangles,
        index_path=args.index_path,
        no_cull_stretch=args.no_cull_stretch
    )