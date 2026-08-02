# Model Zoo

- We provide pre-trained models for view synthesis with 3D Gaussian splatting fine-tuned or trained with geometry losses.

- We assume that the downloaded weights are stored in the `pretrained` directory. It's recommended to create a symbolic link from `YOUR_MODEL_PATH` to `pretrained` using
```
ln -s YOUR_MODEL_PATH pretrained
```

## Gaussian Splatting

- The models are fully trained or fine-tuned on DL3DV (dl3dv) datasets with various geometry loss configurations.

| Model | Params (M) | Download |
| --- | --- | --- |
| small_gggs_nc_full.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gggs_nc_full.ckpt?download=true) |
| small_gggs_nc_fine.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gggs_nc_fine.ckpt?download=true) |
| small_gggs_eanc_full.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gggs_eanc_full.ckpt?download=true) |
| small_gggs_eanc_fine.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gggs_eanc_fine.ckpt?download=true) |
| base_gggs_eanc_fine.ckpt | 223 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/base_gggs_eanc_fine.ckpt?download=true) |
| small_gof_nc_full.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gof_nc_full.ckpt?download=true) |
| small_gof_nc_fine.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gof_nc_fine.ckpt?download=true) |
| small_gof_eanc_full.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gof_eanc_full.ckpt?download=true) |
| small_gof_eanc_fine.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_gof_eanc_fine.ckpt?download=true) |
| base_gof_eanc_fine.ckpt | 223 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/base_gof_eanc_fine.ckpt?download=true) |


## Gaussian Splatting + Mesh NVS

- These models are fine-tuned from the small_gggs_eanc_fine checkpoint to output meshes with increased visual quality.

| Model | Params (M) | Download |
| --- | --- | --- |
| small_mesh.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_mesh.ckpt?download=true) |
| small_mesh_eanc_gauss.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_mesh_eanc_gauss.ckpt?download=true) |
| small_mesh_eanc_gauss_mesh.ckpt | 76 | [download](https://huggingface.co/blancocd/Geometrically_Accurate_FeedForward_Gaussian_Splatting_for_Unbounded_Scenes/resolve/main/small_mesh_eanc_gauss_mesh.ckpt?download=true) |