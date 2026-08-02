from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Literal

try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..geometry.projection import sample_image_grid, get_camera_rays, pcd_to_normal, depth_to_normal, get_valid_normal_mask
from ..geometry.img_warp import warp_with_depth_pose
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor, viz_diff_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply
from ..visualization.export_point_cloud import export_to_point_cloud, transform_points
from ..evaluation.depth_metrics import compute_depth_errors, compute_rend_depth_errors
from ..loss.loss_depth_smooth import get_smooth_loss, get_image_edge_weight_map
from ..mesh.initialize import initialize_mesh, initialize_mesh_vertices, convert_vertices_to_meshes
from ..mesh import MeshRasterizer, MeshRenderer, RasterizationSettings


from .types import Gaussians

try:
    from bitsandbytes.optim import AdamW8bit
except:
    pass

slurm_id_logged = False


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    lr_depth: float
    weight_decay: float
    warm_up_ratio: float
    adamw_8bit: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_high_res_image: bool
    save_high_res_mesh_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    save_gaussian_npz: bool
    no_align_to_view: bool
    save_point_cloud: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int
    render_input_views: bool
    render_both_views: bool
    inference_window_size: int | None
    profile_model: bool
    test_zero_order_sh_only: bool

    save_rendered_depth: bool
    save_depth_diff: bool
    save_depth_normal: bool
    save_gt_depth_normal: bool
    save_depth_normal_diff: bool

    save_mesh: bool
    save_mesh_renders: bool
    opacity_threshold: float
    num_binary_steps: int
    vertex_corner_type: Literal['box', 'tetrahedron', 'octahedron', 'center_only']
    triangle_sigma: float


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    eval_depth: bool
    train_ignore_large_loss: float
    no_log_projections: bool

    log_depth_loss: bool
    depth_smooth_loss_weight: float
    depth_smooth_loss_nonorm: bool

    normal_cons_loss_weight: float
    edge_aware_normal_cons: bool

    img_warp_loss_weight: float
    edge_aware_img_warp: bool
    alpha_edge_aware_img_warp: float

    mesh_nvs_weight: float
    opacity_threshold: float
    num_binary_steps: int
    triangle_sigma: float
    remove_duplicates: bool
    vertex_corner_type: Literal['box', 'tetrahedron', 'octahedron', 'center_only']

    mesh_normal_cons_loss_weight: float

    no_log_video: bool

    # when doing refinement, supervise input view or not since we also render input views
    loss_on_input_views: bool

    # half res lpips loss to save memory
    half_res_lpips_loss: bool

    # local window training
    train_window_size: int | None

    # visualizations in validation step
    viz_geo_renders: bool
    viz_geo_renders_gaussians: bool
    viz_img_warp: bool

@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    mesh_decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        mesh_decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.mesh_decoder = mesh_decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0
        self.skipped_count = 0

        if self.decoder.cfg.name in ['gof', 'gggs']:
            self.encoder.cfg.render_with_quats = True

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0, "extraction": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["context"]["image"].shape
        _, t, _, _, _ = batch["target"]["image"].shape
        # Run the model.
        if self.train_cfg.train_window_size is not None:
            assert self.train_cfg.train_window_size > 0
            window = self.train_cfg.train_window_size
            total_views = batch["context"]["image"].size(1)

            window_indices = sliding_window_indices(total_views, window, 0)

            all_gaussians = []
            all_states = []  # for global refinement

            # sliding window inference
            for indices in window_indices:
                start, end = indices
                curr_window_input = {
                    "extrinsics": batch["context"]["extrinsics"][:, start:end],
                    "intrinsics": batch["context"]["intrinsics"][:, start:end],
                    "image": batch["context"]["image"][:, start:end],
                    "near": batch["context"]["near"][:, start:end],
                    "far": batch["context"]["far"][:, start:end],
                    "index": batch["context"]["index"][:, start:end],
                }

                if self.encoder.cfg.num_refine > 0:
                    with torch.no_grad():
                        curr_gaussians = self.encoder(
                            curr_window_input,
                            self.global_step,
                            deterministic=False,
                        )
                else:
                    curr_gaussians = self.encoder(
                        curr_window_input,
                        self.global_step,
                        deterministic=False,
                    )

                if isinstance(curr_gaussians, dict):
                    pred_depths = curr_gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in curr_gaussians:
                        condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                    else:
                        condition_features = None
                    curr_gaussians = curr_gaussians["gaussians"]

                all_gaussians.append(curr_gaussians)
                all_states.append(condition_features)

            # merge all gaussians
            # ['means', 'covariances', 'harmonics', 'opacities']
            gaussians = Gaussians(
                torch.cat([g.means for g in all_gaussians], dim=1),
                torch.cat([g.covariances for g in all_gaussians], dim=1),
                torch.cat([g.harmonics for g in all_gaussians], dim=1),
                torch.cat([g.opacities for g in all_gaussians], dim=1),
                scales=torch.cat([g.scales for g in all_gaussians], dim=1),
                rotations=torch.cat([g.rotations for g in all_gaussians], dim=1),
                rotations_unnorm=torch.cat([g.rotations_unnorm for g in all_gaussians], dim=1),
            )

            # global refine after simply combining local window gaussians
            if self.encoder.cfg.num_refine > 0:
                # combine all condition features
                out = []
                is_ori_feature = True
                for i in range(len(all_states)):
                    curr = all_states[i]
                    if curr.dim() == 4:  
                        # [BV, C, H, W]
                        curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                        is_ori_feature = True
                    elif curr.dim() == 2:
                        # [BVHW, C]
                        curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                        h=h // self.encoder.cfg.latent_downsample,
                        w=w // self.encoder.cfg.latent_downsample,
                        )
                        is_ori_feature = False
                    else:
                        raise NotImplementedError

                    out.append(curr)

                # concat
                if is_ori_feature:
                    concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                    concat = rearrange(concat, "b v c h w -> (b v) c h w")
                else:
                    concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                    concat = rearrange(concat, "b v h w c -> (b v) c h w")

                condition_features = concat

        else:
            if self.encoder.cfg.num_refine > 0:
                with torch.no_grad():
                    gaussians = self.encoder(
                        batch["context"], self.global_step, False, scene_names=batch["scene"]
                    )
            else:
                gaussians = self.encoder(
                    batch["context"], self.global_step, False, scene_names=batch["scene"]
                )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if self.encoder.cfg.num_refine > 0:
                    condition_features = gaussians["condition_features"]
                gaussians = gaussians["gaussians"]

        supervise_intermediate_depth = False

        if self.encoder.cfg.num_refine > 0:
            refine_output = self.encoder.forward_update(
                batch["context"],
                batch["target"],
                condition_features,
                gaussians,
                self.decoder,
                batch["context_remain"] if "context_remain" in batch.keys() else None,
            )

            render_output = refine_output['render']
            gaussian_output = refine_output['gaussian']

            if self.encoder.cfg.num_refine == 0:
                init_output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

                render_output.insert(0, init_output)
                gaussian_output.insert(0, gaussians)

            render_input_views = refine_output['render_input']

            delta_means = refine_output['delta_means']
            delta_scales = refine_output['delta_scales']

            # the last output
            tgt_output = render_output[-1]
            gaussians = gaussian_output[-1]

        else:
            if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
                supervise_intermediate_depth = True
                assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
                num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                    0
                )
                # add loss to intermediate depth predictions
                target_extrinsics = torch.cat(
                    [batch["target"]["extrinsics"]] * num_depths, dim=0
                )
                target_intrinsics = torch.cat(
                    [batch["target"]["intrinsics"]] * num_depths, dim=0
                )
                target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
                target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

                output_all = self.decoder.forward(
                    gaussians,
                    target_extrinsics,
                    target_intrinsics,
                    target_near,
                    target_far,
                    (h, w),
                    depth_mode=None,
                )
                # split
                batch_size = batch["target"]["extrinsics"].size(0)
                # order: intermediate depth, final depth
                output_intermediate = DecoderOutput(
                    color=output_all.color[:-batch_size],
                    depth=(
                        output_all.depth[:-batch_size]
                        if output_all.depth is not None
                        else None
                    ),
                )
                tgt_output = DecoderOutput(
                    color=output_all.color[-batch_size:],
                    depth=(
                        output_all.depth[-batch_size:]
                        if output_all.depth is not None
                        else None
                    ),
                )
            else:
                empty_mesh = False
                if self.train_cfg.mesh_nvs_weight != 0.:
                    mesh_vertices = initialize_mesh_vertices(gaussians, 
                                    batch["context"]["extrinsics"], batch["context"]["intrinsics"],
                                    batch["context"]["near"], batch["context"]["far"], 
                                    (h, w), 'gggs', self.train_cfg.opacity_threshold, self.train_cfg.num_binary_steps, use_gpu_del=True,
                                    remove_duplicates=self.train_cfg.remove_duplicates, vertex_corner_type=self.train_cfg.vertex_corner_type)
                    is_empty = any(v.shape[0] == 0 for v in mesh_vertices.positions) or \
                            any(f.shape[0] == 0 for f in mesh_vertices.faces)
                            
                    if is_empty:
                        empty_mesh = True
                    else:
                        tgt_output_mesh = self.mesh_decoder.forward(
                            mesh_vertices,
                            batch["target"]["extrinsics"],
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (h, w),
                            sigma=self.train_cfg.triangle_sigma,
                        )
                        ctx_output_mesh = self.mesh_decoder.forward(
                            mesh_vertices,
                            batch["context"]["extrinsics"],
                            batch["context"]["intrinsics"],
                            batch["context"]["near"],
                            batch["context"]["far"],
                            (h, w),
                            sigma=self.train_cfg.triangle_sigma,
                        )
                tgt_output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )
                ctx_output = self.decoder.forward(
                    gaussians,
                    batch["context"]["extrinsics"],
                    batch["context"]["intrinsics"],
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (h, w),
                    depth_mode=None,
                )

        target_gt = batch["target"]["image"]

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        # loss with refinement
        if self.encoder.cfg.num_refine > 0:
            num_output = len(render_output)
            for i in range(num_output):
                # Compute training metrics.
                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(render_output[i].color, "b v c h w -> (b v) c h w"),
                )
                self.log(f"train/psnr_{i}", psnr_probabilistic.mean())

                # compute losses
                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            render_output[i],
                            batch,
                            None,
                            self.global_step,

                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            render_output[i],
                            batch,
                            None,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    self.log(f"loss/{loss_fn.name}_{i}", loss)

                    curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                                num_output - 1 - i
                            )

                    total_loss = total_loss + curr_loss_weight * loss

            # loss on input views
            if self.train_cfg.loss_on_input_views:
                for i in range(num_output):
                    # Compute training metrics.
                    psnr_probabilistic = compute_psnr(
                        rearrange(batch["context"]["image"], "b v c h w -> (b v) c h w"),
                        rearrange(render_input_views[i].color, "b v c h w -> (b v) c h w"),
                    )
                    self.log(f"train/input_psnr_{i}", psnr_probabilistic.mean())

                    # compute losses
                    for loss_fn in self.losses:
                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                render_input_views[i],
                                batch,
                                gaussian_output[i],
                                self.global_step,
    
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                                loss_on_input_views=True,
                            )
                        else:
                            loss = loss_fn.forward(
                                render_input_views[i],
                                batch,
                                gaussian_output[i],
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                                loss_on_input_views=True,
                                half_res_lpips=self.train_cfg.half_res_lpips_loss,
                            )
                        self.log(f"loss/input_{loss_fn.name}_{i}", loss)

                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                                    num_output - 1 - i
                                )

                        total_loss = total_loss + curr_loss_weight * loss

        else:
            gaussian_nvs_weight = 1.
            if self.train_cfg.mesh_nvs_weight != 0. and not empty_mesh:
                if self.train_cfg.mesh_nvs_weight < 0.:
                    gaussian_nvs_weight = 0
                else:
                    gaussian_nvs_weight = 1 / self.train_cfg.mesh_nvs_weight

                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(tgt_output_mesh.color, "b v c h w -> (b v) c h w"),
                )
                self.log("train/psnr", psnr_probabilistic.mean())

                # Compute metrics.
                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(tgt_output.color, "b v c h w -> (b v) c h w"),
                )
                self.log("train/gaussians_psnr", psnr_probabilistic.mean())

                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            tgt_output_mesh,
                            batch,
                            gaussians,
                            self.global_step,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            tgt_output_mesh,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    self.log(f"loss/{loss_fn.name}", loss)
                    total_loss = total_loss + loss

                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            tgt_output,
                            batch,
                            gaussians,
                            self.global_step,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            tgt_output,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    loss = gaussian_nvs_weight * loss
                    self.log(f"loss/gaussians_{loss_fn.name}", loss)
                    total_loss = total_loss + loss

            else:
                # Compute metrics.
                psnr_probabilistic = compute_psnr(
                    rearrange(target_gt, "b v c h w -> (b v) c h w"),
                    rearrange(tgt_output.color, "b v c h w -> (b v) c h w"),
                )
                self.log("train/psnr", psnr_probabilistic.mean())
                    
                for loss_fn in self.losses:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            tgt_output,
                            batch,
                            gaussians,
                            self.global_step,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            tgt_output,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                            half_res_lpips=self.train_cfg.half_res_lpips_loss,
                        )
                    loss = gaussian_nvs_weight * loss
                    self.log(f"loss/{loss_fn.name}", loss)
                    total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
    
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,

                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        # depth smooth loss
        if self.train_cfg.depth_smooth_loss_weight > 0:
            imgs = batch["context"]["image"].flatten(0, 1)  # [BV, 3, H, W]
            
            if supervise_intermediate_depth:
                assert pred_depths.size(0) % batch_size == 0
                num_depths = pred_depths.size(0) // batch_size
                depth_smooth_loss = 0
                for i in range(num_depths):
                    curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                        num_depths - i - 1)

                    depth = pred_depths[(batch_size * i):(
                            batch_size * (i + 1))].flatten(0, 1).unsqueeze(1)  # [BV, 1, H, W]
                    disp = 1. / depth
                    if self.train_cfg.depth_smooth_loss_nonorm:
                        norm_disp = disp
                    else:
                        mean_disp = disp.mean(2, True).mean(3, True)
                        norm_disp = disp / (mean_disp + 1e-7)

                    # resize to depth's resolution
                    if imgs.shape[-2:] != norm_disp.shape[-2:]:
                        imgs = F.interpolate(imgs, size=norm_disp.shape[-2:], mode='bilinear', align_corners=True)

                    depth_smooth_loss = get_smooth_loss(norm_disp, imgs)

                    depth_smooth_loss = depth_smooth_loss + curr_loss_weight * depth_smooth_loss

            else:
                depth = pred_depths.flatten(0, 1).unsqueeze(1)

                disp = 1. / depth
                if self.train_cfg.depth_smooth_loss_nonorm:
                    norm_disp = disp
                else:
                    mean_disp = disp.mean(2, True).mean(3, True)
                    norm_disp = disp / (mean_disp + 1e-7)

                # resize to depth's resolution
                if imgs.shape[-2:] != norm_disp.shape[-2:]:
                    imgs = F.interpolate(imgs, size=norm_disp.shape[-2:], mode='bilinear', align_corners=True)

                depth_smooth_loss = get_smooth_loss(norm_disp, imgs)

            depth_smooth_loss = self.train_cfg.depth_smooth_loss_weight * depth_smooth_loss

            self.log(f"loss/depth_smooth", depth_smooth_loss)
            total_loss = total_loss + depth_smooth_loss

        # as it tests depth generalization it uses rendered depth from tgt camera poses 
        if self.train_cfg.img_warp_loss_weight > 0:
            context_imgs = batch["context"]["image"]  # [B, V, 3, H, W]
            
            # Unnormalize intrinsics
            context_intrinsics = batch["context"]["intrinsics"].clone()  # [B, V, 3, 3]
            context_intrinsics[:, :, 0] *= w
            context_intrinsics[:, :, 1] *= h
            context_extrinsics = batch["context"]["extrinsics"]  # [B, V, 4, 4]

            # rendered depths
            render_target_depths = tgt_output.depth # [B, T, H, W]
            render_context_depths = ctx_output.depth  # [B, V, H, W]

            target_imgs = batch["target"]["image"]  # [B, T, 3, H, W]
            target_intrinsics = batch["target"]["intrinsics"].clone()  # [B, T, 3, 3]
            # Unnormalize intrinsics
            target_intrinsics[:, :, 0] *= w
            target_intrinsics[:, :, 1] *= h
            target_extrinsics = batch["target"]["extrinsics"]  # [B, T, 4, 4]

            context_inv = context_extrinsics.unsqueeze(1).inverse()  
            target2context = torch.matmul(context_inv, target_extrinsics.unsqueeze(2))

            # Assume the intrinsics are the same for all images in scene, true for DL3DV
            intrinsics = target_intrinsics.unsqueeze(2).expand(-1, -1, v, -1, -1)  # [B, T, V, 3, 3]

            context_imgs = context_imgs.unsqueeze(1).expand(-1, t, -1, -1, -1, -1)  # [B, T, V, 3, H, W]
            render_target_depths = render_target_depths.unsqueeze(2).expand(-1, -1, v, -1, -1)  # [B, T, V, H, W]
            render_context_depths = render_context_depths.unsqueeze(1).expand(-1, t, -1, -1, -1)  # [B, T, V, H, W]

            target_imgs = target_imgs.unsqueeze(2).expand(-1, -1, v, -1, -1, -1)  # [B, T, V, 3, H, W]
            target_imgs_flat = rearrange(target_imgs, "b t v c x y -> (b t v) c x y")  # [BTV, 3, H, W]

            weights_flat = None
            if self.train_cfg.edge_aware_img_warp:
                with torch.no_grad():
                    img_edge_weight_maps = get_image_edge_weight_map(batch["target"]["image"], 
                                                                        self.train_cfg.alpha_edge_aware_img_warp)
                    img_edge_weight_maps = 2.0 - img_edge_weight_maps # [B, T, H, W]
                    
                    weights = img_edge_weight_maps.unsqueeze(2).expand(-1, -1, v, -1, -1)
                    weights_flat = rearrange(weights, "b t v h w -> (b t v) h w")

            def quick_warp_loss(c_imgs, intr, t2c, t_depth, c_depth, t_imgs, w_flat=None):
                warped, masks = warp_with_depth_pose(
                    c_imgs, intr, t2c, t_depth, depth_tgt=c_depth, return_mask=True
                )
                # [BTV, 3, H, W] -> [BTV, H, W]
                ssim_loss = SSIM().to(t_imgs.device)(warped, t_imgs).mean(dim=1)
                l1_loss = (warped - t_imgs).abs().mean(dim=1)

                raw_loss = 0.85 * ssim_loss + 0.15 * l1_loss
                masked_loss = raw_loss[masks]

                if self.train_cfg.edge_aware_img_warp and w_flat is not None:
                    masked_weights = w_flat[masks]
                    return (masked_loss * masked_weights).sum() / masked_weights.sum().clamp(min=1e-6)
                else:
                    return masked_loss.mean()

            from torch.utils.checkpoint import checkpoint
            warp_error = checkpoint(
                quick_warp_loss,
                rearrange(context_imgs, "b t v c h w -> (b t v) c h w"),
                rearrange(intrinsics, "b t v x y -> (b t v) x y"),
                rearrange(target2context, "b t v x y -> (b t v) x y"),
                rearrange(render_target_depths, "b t v h w -> (b t v) h w"),
                rearrange(render_context_depths.detach(), "b t v h w -> (b t v) h w"),
                target_imgs_flat,
                weights_flat,
                use_reentrant=False
            )

            img_warp_loss = self.train_cfg.img_warp_loss_weight * warp_error

            self.log(f"loss/img_warp", img_warp_loss.item())
            total_loss = total_loss + img_warp_loss


        if self.train_cfg.normal_cons_loss_weight > 0:
            # [B, T, 3, H, W]
            normal = tgt_output.normal
            # [B, T, H, W]
            normal_mask = tgt_output.accumulated_alpha > 1e-6
            
            # [B, T, 3, H, W]->[B, T, H, W, 3]->[B, T, H-2, W-2, 3]
            normal = normal.permute(0, 1, 3, 4, 2)
            normal_nomargins = normal[:, :, 1:-1, 1:-1, :]
            # [B, T, H, W]->[B, T, H-2, W-2]
            normal_mask = normal_mask[:, :, 1:-1, 1:-1]

            # [B, T, H-2, W-2, 3]
            depth_normal, depth_normal_mask = depth_to_normal(tgt_output.depth, batch["target"]["intrinsics"], h, w)
            # [B, T, H-2, W-2]
            mask = normal_mask & depth_normal_mask

            if mask.any():
                # [B, T, H-2, W-2, 3]->[*, 3]->[*]
                normal_cons_map = 1 - (depth_normal[mask] * normal_nomargins[mask]).sum(dim=-1)

                if self.train_cfg.edge_aware_normal_cons:
                    with torch.no_grad():
                        img_edge_weight_maps = get_image_edge_weight_map(batch["target"]["image"]) # [B, T, H, W]
                        img_edge_weight_maps = img_edge_weight_maps[:, :, 1:-1, 1:-1] # [B, T, H-2, W-2]
                    # [*]
                    normal_cons_map = normal_cons_map * img_edge_weight_maps[mask]

                normal_cons_loss = normal_cons_map.mean()
            else:
                normal_cons_loss = torch.tensor(0.0, device=normal.device)

            normal_cons_loss *= self.train_cfg.normal_cons_loss_weight
            self.log(f"loss/normal_cons_loss", normal_cons_loss)
                
            total_loss = total_loss + normal_cons_loss

        if self.train_cfg.mesh_normal_cons_loss_weight > 0 and self.train_cfg.mesh_nvs_weight != 0. and not empty_mesh:
            assert self.train_cfg.mesh_nvs_weight != 0., "Mesh normal consistency loss only works when mesh_nvs_weight is not 0."
            # [B, T, 3, H, W]
            normal = tgt_output_mesh.normal
            # [B, T, H, W]
            normal_mask = tgt_output_mesh.accumulated_alpha > 1e-6
            
            # [B, T, 3, H, W]->[B, T, H, W, 3]->[B, T, H-2, W-2, 3]
            normal = normal.permute(0, 1, 3, 4, 2)
            normal_nomargins = normal[:, :, 1:-1, 1:-1, :]
            # [B, T, H, W]->[B, T, H-2, W-2]
            normal_mask = normal_mask[:, :, 1:-1, 1:-1]

            # [B, T, H-2, W-2, 3]
            depth_normal, depth_normal_mask = depth_to_normal(tgt_output.depth.detach(), batch["target"]["intrinsics"], h, w)
            # [B, T, H-2, W-2]
            mask = normal_mask & depth_normal_mask

            if mask.any():
                # [B, T, H-2, W-2, 3]->[*, 3]->[*]
                normal_cons_map = 1 - (depth_normal[mask] * normal_nomargins[mask]).sum(dim=-1)

                if self.train_cfg.edge_aware_normal_cons:
                    with torch.no_grad():
                        img_edge_weight_maps = get_image_edge_weight_map(batch["target"]["image"]) # [B, T, H, W]
                        img_edge_weight_maps = img_edge_weight_maps[:, :, 1:-1, 1:-1] # [B, T, H-2, W-2]
                    # [*]
                    normal_cons_map = normal_cons_map * img_edge_weight_maps[mask]

                normal_cons_loss = normal_cons_map.mean()
            else:
                normal_cons_loss = torch.tensor(0.0, device=normal.device)

            normal_cons_loss *= self.train_cfg.mesh_normal_cons_loss_weight
            self.log(f"loss/mesh_normal_cons_loss", normal_cons_loss)
                
            total_loss = total_loss + normal_cons_loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # log gaussians scales
        if self.encoder.cfg.num_refine > 0:
            num_output = len(delta_means)
            # delta means
            if isinstance(delta_means[0], torch.Tensor):
                for i in range(num_output):
                    self.log(f"update{i}/delta_means_min", delta_means[i].abs().min().item())
                    self.log(f"update{i}/delta_means_mean", delta_means[i].abs().mean().item())
                    self.log(f"update{i}/delta_means_max", delta_means[i].abs().max().item())

            # delta scales
            if isinstance(delta_scales[0], torch.Tensor):
                for i in range(num_output):
                    self.log(f"update{i}/delta_scales_min", delta_scales[i].abs().min().item())
                    self.log(f"update{i}/delta_scales_mean", delta_scales[i].abs().mean().item())
                    self.log(f"update{i}/delta_scales_max", delta_scales[i].abs().max().item())

        self.log("info/gaussian_scale_min", gaussians.scales.min().item())
        self.log("info/gaussian_scale_max", gaussians.scales.max().item())
        self.log("info/gaussian_scale_mean", gaussians.scales.mean().item())

        # log gaussians opacities
        self.log("info/gaussian_opacity_min", gaussians.opacities.min().item())
        self.log("info/gaussian_opacity_max", gaussians.opacities.max().item())
        self.log("info/gaussian_opacity_mean", gaussians.opacities.mean().item())

        # log gaussians opacities raw
        self.log("info/gaussian_opacity_raw_min", torch.logit(gaussians.opacities, eps=1e-6).min().item())
        self.log("info/gaussian_opacity_raw_max", torch.logit(gaussians.opacities, eps=1e-6).max().item())
        self.log("info/gaussian_opacity_raw_mean", torch.logit(gaussians.opacities, eps=1e-6).abs().mean().item())

        # log gaussians mean
        self.log("info/gaussian_mean_min", gaussians.means.min().item())
        self.log("info/gaussian_mean_max", gaussians.means.max().item())
        self.log("info/gaussian_mean_mean", gaussians.means.abs().mean().item())

        # log gaussians rotation unnorm
        self.log("info/gaussian_rotation_unnorm_min", gaussians.rotations_unnorm.min().item())
        self.log("info/gaussian_rotation_unnorm_max", gaussians.rotations_unnorm.max().item())
        self.log("info/gaussian_rotation_unnorm_mean", gaussians.rotations_unnorm.abs().mean().item())

        # log gaussians sh
        self.log("info/gaussian_sh_min", gaussians.harmonics.min().item())
        self.log("info/gaussian_sh_max", gaussians.harmonics.max().item())
        self.log("info/gaussian_sh_mean", gaussians.harmonics.abs().mean().item())

        if self.train_cfg.mesh_nvs_weight != 0.:
            # average of number of vertices across list of batches
            avg_num_vertices = sum(v.shape[0] for v in mesh_vertices.positions) / len(mesh_vertices.positions)
            self.log("info/mesh_vertex_count", avg_num_vertices)
            avg_num_faces = sum(f.shape[0] for f in mesh_vertices.faces) / len(mesh_vertices.faces)
            self.log("info/mesh_face_count", avg_num_faces)

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")

        global slurm_id_logged
        if self.global_rank == 0 and not slurm_id_logged:
            print('slurm id:', os.environ.get('SLURM_JOB_ID'))
            slurm_id_logged = True

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        if self.test_cfg.render_input_views:
            # to see how good the model performs on the input views
            b, v, _, h, w = batch["context"]["image"].shape
        elif self.test_cfg.render_both_views:
            b, v, _, h, w = batch["context"]["image"].shape
            _, t, _, _, _ = batch["target"]["image"].shape
            v = v + t
        else:
            b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None
        depth_gt = None

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            if self.test_cfg.inference_window_size is not None:
                assert self.test_cfg.inference_window_size > 0
                window = self.test_cfg.inference_window_size
                total_views = batch["context"]["image"].size(1)

                window_indices = sliding_window_indices(total_views, window, 0)

                all_gaussians = []
                all_states = []  # for global refinement

                # sliding window inference
                for indices in window_indices:
                    start, end = indices
                    curr_window_input = {
                        "extrinsics": batch["context"]["extrinsics"][:, start:end],
                        "intrinsics": batch["context"]["intrinsics"][:, start:end],
                        "image": batch["context"]["image"][:, start:end],
                        "near": batch["context"]["near"][:, start:end],
                        "far": batch["context"]["far"][:, start:end],
                        "index": batch["context"]["index"][:, start:end],
                    }

                    curr_gaussians = self.encoder(
                        curr_window_input,
                        self.global_step,
                        deterministic=False,
                    )

                    if isinstance(curr_gaussians, dict):
                        pred_depths = curr_gaussians["depths"]
                        if "depth" in batch["context"]:
                            depth_gt = batch["context"]["depth"]
                        if "condition_features" in curr_gaussians:
                            condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                        else:
                            condition_features = None
                        curr_gaussians = curr_gaussians["gaussians"]

                    all_gaussians.append(curr_gaussians)
                    all_states.append(condition_features)

                # merge all gaussians
                # ['means', 'covariances', 'harmonics', 'opacities']
                gaussians = Gaussians(
                    torch.cat([g.means for g in all_gaussians], dim=1),
                    torch.cat([g.covariances for g in all_gaussians], dim=1),
                    torch.cat([g.harmonics for g in all_gaussians], dim=1),
                    torch.cat([g.opacities for g in all_gaussians], dim=1),
                    scales=torch.cat([g.scales for g in all_gaussians], dim=1),
                    rotations=torch.cat([g.rotations for g in all_gaussians], dim=1),
                    rotations_unnorm=torch.cat([g.rotations_unnorm for g in all_gaussians], dim=1),
                )

                # global refine after simply combining local window gaussians
                if self.encoder.cfg.num_refine > 0:
                    # combine all condition features
                    out = []
                    is_ori_feature = False
                    for i in range(len(all_states)):
                        curr = all_states[i]
                        if curr.dim() == 4:  
                            # [BV, C, H, W]
                            curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                            is_ori_feature = True
                        elif curr.dim() == 2:
                            # [BVHW, C]
                            curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                            h=h // self.encoder.cfg.latent_downsample,
                            w=w // self.encoder.cfg.latent_downsample,
                            )
                            is_ori_feature = False
                        else:
                            raise NotImplementedError

                        out.append(curr)

                    # concat
                    if is_ori_feature:
                        concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                        concat = rearrange(concat, "b v c h w -> (b v) c h w")
                    else:
                        concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                        concat = rearrange(concat, "b v h w c -> (b v) c h w")

                    refine_output = self.encoder.forward_update(
                            batch["context"],
                            batch["target"],
                            concat,
                            gaussians,
                            self.decoder,
                            batch["context_remain"] if "context_remain" in batch.keys() else None,
                        )
                        
                    render_output = refine_output['render']
                    gaussians = refine_output['gaussian'][-1]

                    output = render_output[-1]

            else:
                if self.encoder.cfg.no_crop_image:
                    # resize the context image
                    num_view, _, ori_h, ori_w = batch["context"]["image"].shape[1:]
                    resize_h, resize_w = int(np.ceil(ori_h / 64)) * 64, int(np.ceil(ori_w / 64)) * 64
                    if ori_h != resize_h or ori_w != resize_w:
                        batch["context"]["image"] = F.interpolate(
                            batch["context"]["image"].flatten(0, 1), size=(resize_h, resize_w), mode='bilinear', align_corners=True
                        ).view(1, num_view, 3, resize_h, resize_w)

                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                    visualization_dump=visualization_dump,
                )

                if isinstance(gaussians, dict):
                    pred_depths = gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in gaussians:
                        condition_features = gaussians["condition_features"]
                    gaussians = gaussians["gaussians"]

                # refine
                if self.encoder.cfg.num_refine > 0:
                    refine_output = self.encoder.forward_update(
                        batch["context"],
                        batch["target"],
                        condition_features,
                        gaussians,
                        self.decoder,
                        batch["context_remain"] if "context_remain" in batch.keys() else None,
                    )
                    render_output = refine_output['render']
                    gaussians = refine_output['gaussian'][-1]

                    output = render_output[-1]

        # save gaussians
        if self.test_cfg.save_gaussian:
            scene = batch["scene"][0]
            save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path, no_align_to_view=self.test_cfg.no_align_to_view,
                              save_gaussian_npz=self.test_cfg.save_gaussian_npz,
                              )

        # test zero-order sh only
        if self.test_cfg.test_zero_order_sh_only:
            gaussians.harmonics[:, :, :, 1:] = 0.

        # save point cloud
        if self.test_cfg.save_point_cloud:
            point_cloud = gaussians.means.reshape(-1, 3).detach()  # [N, 3]

            point_cloud = point_cloud.cpu().numpy()

            colors = batch["context"]["image"][0].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()

            scene = batch["scene"][0]
            save_path = get_cfg()['output_dir'] + '/pointcloud/' + f"{scene}.ply"

            export_to_point_cloud(point_cloud, colors,
                save_path=save_path
                )

        if self.test_cfg.save_mesh:
            decoder_name = self.decoder.cfg.name
            decoder_name = decoder_name if decoder_name in ['gof', 'gggs'] else 'gof'
            with self.benchmarker.time("extraction"):
                mesh = initialize_mesh_vertices(
                    gaussians, batch["context"]["extrinsics"], batch["context"]["intrinsics"],
                    batch["context"]["near"], batch["context"]["far"], (h, w),
                    decoder_name, self.test_cfg.opacity_threshold, self.test_cfg.num_binary_steps, use_gpu_del=True,
                    remove_duplicates=False, vertex_corner_type=self.test_cfg.vertex_corner_type
                )

            verts = mesh.positions[0]
            faces = mesh.faces[0]
            sh = mesh.harmonics[0]

            SH_C0 = 0.28209479177387814
            sh_dc = sh[:, 0, :]
            
            vertex_colors = (sh_dc * SH_C0 + 0.5).clamp(min=0.0, max=1.0)
            vertex_colors = (vertex_colors * 255.0).to(torch.uint8)

            save_dir = Path(get_cfg()['output_dir']) / 'mesh'
            os.makedirs(save_dir, exist_ok=True)
            
            trimesh_mesh = trimesh.Trimesh(
                vertices=verts.cpu().numpy(), 
                faces=faces.cpu().numpy(), 
                vertex_colors=vertex_colors.cpu().numpy(),
                process=False
            )

            scene = batch["scene"][0]
            trimesh_mesh.remove_unreferenced_vertices()            
            trimesh_mesh.export(save_dir / (scene + '.ply'))

        with self.benchmarker.time("decoder", num_calls=v):
            if self.test_cfg.render_both_views:
                rgb_gt = torch.cat((batch['context']["image"][0], batch['target']["image"][0]), dim=0)
                depth_gt = torch.cat((batch['context']["depth"][0], batch['target']["depth"][0]), dim=0) if "depth" in batch['context'] else None
                indices = torch.cat((batch['context']["index"][0], batch['target']["index"][0]), dim=0)
                camera_poses = torch.cat((batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0]), dim=0)
                intrinsics = torch.cat((batch['context']["intrinsics"][0], batch['target']["intrinsics"][0]), dim=0)
                near = torch.cat((batch['context']["near"][0], batch['target']["near"][0]), dim=0)
                far = torch.cat((batch['context']["far"][0], batch['target']["far"][0]), dim=0)
            else:
                split = 'context' if self.test_cfg.render_input_views else 'target'
                rgb_gt = batch[split]["image"][0]                   # [V, 3, H, W]
                depth_gt = batch[split]["depth"][0] if "depth" in batch[split] else None  # [V, H, W]
                indices = batch[split]["index"][0]                  # [V]
                camera_poses = batch[split]["extrinsics"][0]      # [V, 4, 4]
                intrinsics = batch[split]["intrinsics"][0]    # [V, 3, 3]
                near = batch[split]["near"][0]              # [V]
                far = batch[split]["far"][0]                # [V]

            if self.test_cfg.stablize_camera:
                stable_poses = render_stabilization_path(
                    camera_poses[0].detach().cpu().numpy(),
                    k_size=self.test_cfg.stab_camera_kernel,
                )

                stable_poses = list(
                    map(
                        lambda x: np.concatenate(
                            (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                        ),
                        stable_poses,
                    )
                )
                stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                    camera_poses
                )
                camera_poses = stable_poses.unsqueeze(0)

            if self.test_cfg.render_chunk_size is not None:
                chunk_size = self.test_cfg.render_chunk_size
                num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                output = None
                for i in range(num_chunks):
                    start = chunk_size * i
                    end = chunk_size * (i + 1)

                    curr_output = self.decoder.forward(
                        gaussians,
                        camera_poses.unsqueeze(0)[:, start:end],
                        intrinsics.unsqueeze(0)[:, start:end],
                        near.unsqueeze(0)[:, start:end],
                        far.unsqueeze(0)[:, start:end],
                        (h, w),
                        depth_mode=None,
                    )

                    if i == 0:
                        output = curr_output
                    else:
                        # ignore depth
                        output.color = torch.cat(
                            (output.color, curr_output.color), dim=1
                        )

            else:
                output = self.decoder.forward(
                    gaussians,
                    camera_poses.unsqueeze(0),
                    intrinsics.unsqueeze(0),
                    near.unsqueeze(0),
                    far.unsqueeze(0),
                    (h, w),
                    depth_mode=None,
                )

                if self.test_cfg.save_mesh_renders:
                    output_mesh = self.mesh_decoder.forward(
                        mesh,
                        camera_poses.unsqueeze(0),
                        intrinsics.unsqueeze(0),
                        near.unsqueeze(0),
                        far.unsqueeze(0),
                        (h, w),
                        sigma=self.test_cfg.triangle_sigma,
                    )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            depth = (
                visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
            )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.detach().cpu().numpy()
                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

        images_prob = output.color[0]
        # Save images.
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(
                    indices, images_prob, rgb_gt
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(indices, images_prob):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")

            if self.test_cfg.save_mesh_renders:
                for index, color in zip(indices, output_mesh.color[0]):
                    save_image(color, path / "images" / scene / f"mesh_color/{index:0>6}.png")

        # Save images.
        if self.test_cfg.save_high_res_image:
            for scale_resolution in [2, 4]:
                curr_output = self.decoder.forward(
                    gaussians,
                    camera_poses.unsqueeze(0),
                    intrinsics.unsqueeze(0),
                    near.unsqueeze(0),
                    far.unsqueeze(0),
                    (h*scale_resolution, w*scale_resolution),
                    depth_mode=None,
                )
                curr_color = curr_output.color[0]
                for index, color in zip(indices, curr_color):
                    save_image(color, path / "images" / scene / f"color_{scale_resolution}/{index:0>6}.png") 

        if self.test_cfg.save_high_res_mesh_image:
            for scale_resolution in [2, 4]:
                curr_output = self.mesh_decoder.forward(
                    mesh,
                    camera_poses.unsqueeze(0),
                    intrinsics.unsqueeze(0),
                    near.unsqueeze(0),
                    far.unsqueeze(0),
                    (h*scale_resolution, w*scale_resolution),
                    sigma=self.test_cfg.triangle_sigma,
                )
                curr_color = curr_output.color[0]
                for index, color in zip(indices, curr_color):
                    save_image(color, path / "images" / scene / f"mesh_color_{scale_resolution}/{index:0>6}.png") 


        if self.test_cfg.save_rendered_depth:
            if output.depth is not None:
                depths = output.depth[0] # [V, H, W]
                depths = rearrange(depths, "v h w -> (v h) w")
                depths = depths.detach().cpu()
                depth_vizs = viz_depth_tensor(
                    1.0 / depths, return_numpy=True
                )  # [H, V*W, 3]
                depth_vizs = rearrange(depth_vizs, "(v h) w c-> v h w c", v=v)

                save_dir = path / "images" / scene / "rendered_depth"
                os.makedirs(save_dir, exist_ok=True)
                for index, depth_viz in zip(indices, depth_vizs):
                    save_path = save_dir / f"{index:0>6}.png"
                    Image.fromarray(depth_viz).save(save_path)
                
                depths = output.depth[0].detach().cpu().numpy()
                for index, depth in zip(indices, depths):
                    save_path = save_dir / f"{index:0>6}.npy"
                    np.save(save_path, depth)
            else:
                print("Depth was not rendered and thus can't be saved")

        # blue is for negative values and red is for positive values
        # blue: rend depth < gt depth, rendered depth is closer than should be
        # red: rend_depth > gt depth, rendered depth is farther than should be
        if self.test_cfg.save_depth_diff:
            if output.depth is not None and depth_gt is not None:
                diff_depths = output.depth[0] - depth_gt # [V, H, W]
                valid_mask = (output.depth[0] > 0) & (depth_gt > 0)
                
                diff_depths = rearrange(diff_depths, "v h w -> (v h) w")
                diff_depths = diff_depths.detach().cpu()
                diff_depths_vizs, cb_fig = viz_diff_tensor(
                    diff_depths, return_numpy=True, return_cb_fig=True
                )  # [H, V*W, 3]
                diff_depths_vizs = rearrange(diff_depths_vizs, "(v h) w c -> v h w c", v=v)
                diff_depths_vizs *= valid_mask[..., None].cpu().numpy() # set invalid pixel to black

                save_dir = path / "images" / scene / "depth_diff"
                os.makedirs(save_dir, exist_ok=True)
                for index, diff_depths_viz in zip(indices, diff_depths_vizs):
                    save_path = save_dir / f"{index:0>6}.png"
                    Image.fromarray(diff_depths_viz).save(save_path)

                cb_fig.savefig(save_dir / "colorbar.png", bbox_inches='tight')
                plt.close(cb_fig)
            else:
                print("Depth was not rendered or GT depth is not provided")

        if self.test_cfg.save_depth_normal:
            if output.depth is not None:
                depths = output.depth[0]
                xy_ray, _ = sample_image_grid((h, w), intrinsics.device)
                save_dir = path / "images" / scene / "depth_normal"
                os.makedirs(save_dir, exist_ok=True)
                for index, depth, intr in zip(indices, depths, intrinsics):
                    # [H, W, 3]
                    directions = get_camera_rays(xy_ray, intr)
                    point_cloud = directions * depth[..., None]

                    # [1, 1, H, W, 3] as we are using the batch-views function
                    point_cloud = point_cloud.unsqueeze(0).unsqueeze(0)
                    # [1, 1, H-2, W-2, 3], [1, 1, H-2, W-2]
                    depth_normal = pcd_to_normal(point_cloud)
                    depth_normal_mask = get_valid_normal_mask(point_cloud)
                    depth_normal *= depth_normal_mask[..., None]
                    depth_normal = depth_normal[0,0]
                    depth_normal_viz = ((depth_normal + 1.0) / 2.0 * 255).cpu().numpy().astype(np.uint8)
                    Image.fromarray(depth_normal_viz).save(save_dir / f"{index:0>6}.png")
            else:
                print("Depth was not rendered and thus depth normal can't be saved")

        if self.test_cfg.save_gt_depth_normal:
            if depth_gt is not None:
                xy_ray, _ = sample_image_grid((h, w), intrinsics.device)
                save_dir = path / "images" / scene / "gt_depth_normal"
                os.makedirs(save_dir, exist_ok=True)
                for index, depth, intr in zip(indices, depth_gt, intrinsics):
                    # [H, W, 3]
                    directions = get_camera_rays(xy_ray, intr)
                    point_cloud = directions * depth[..., None]

                    # [1, 1, H, W, 3] as we are using the batch-views function
                    point_cloud = point_cloud.unsqueeze(0).unsqueeze(0)
                    # [1, 1, H-2, W-2, 3], [1, 1, H-2, W-2]
                    depth_normal = pcd_to_normal(point_cloud)
                    depth_normal_mask = get_valid_normal_mask(point_cloud)
                    depth_normal *= depth_normal_mask[..., None]
                    
                    depth_normal = depth_normal[0,0]
                    depth_normal_viz = ((depth_normal + 1.0) / 2.0 * 255).cpu().numpy().astype(np.uint8)
                    Image.fromarray(depth_normal_viz).save(save_dir / f"{index:0>6}.png")
            else:
                print("GT Depth is not provided and thus depth normal can't be saved")

        if self.test_cfg.save_depth_normal_diff:
            if output.depth is not None and depth_gt is not None:
                depths = output.depth # [1, V, H, W]
                gt_depth = depth_gt.unsqueeze(0) # [1, V, H, W]

                xy_ray, _ = sample_image_grid((h, w), intrinsics.device)
                save_dir = path / "images" / scene / "depth_normal_diff"
                os.makedirs(save_dir, exist_ok=True)

                # [1, V, H-2, W-2, 3], [1, V, H-2, W-2]
                depth_normal, depth_normal_mask = depth_to_normal(depths, intrinsics.unsqueeze(0), h, w)
                gt_depth_normal, gt_depth_normal_mask = depth_to_normal(gt_depth, intrinsics.unsqueeze(0), h, w)
                valid_mask = depth_normal_mask & gt_depth_normal_mask
                
                # dot prod range is 1 to -1, better to worse. [V, H-2, W-2]
                depth_normal_diff = (depth_normal*gt_depth_normal)[0].sum(dim=-1)
                depth_normal_diff = torch.rad2deg(torch.acos(depth_normal_diff.clamp(-1.0 + 1e-7, 1.0 - 1e-7)))

                depth_normal_diff = rearrange(depth_normal_diff, "v h w -> (v h) w")
                depth_normal_diff = depth_normal_diff.detach().cpu()
                depth_normal_diff_vizs, cb_fig = viz_depth_tensor(
                    depth_normal_diff, return_numpy=True, return_cb_fig=True
                )  # [H, V*W, 3]
                depth_normal_diff_vizs = rearrange(depth_normal_diff_vizs, "(v h) w c -> v h w c", v=v)
                depth_normal_diff_vizs *= valid_mask[0, ..., None].cpu().numpy()

                for index, depth_normal_diff_viz in zip(indices, depth_normal_diff_vizs):
                    save_path = save_dir / f"{index:0>6}.png"
                    Image.fromarray(depth_normal_diff_viz).save(save_path)

                cb_fig.savefig(save_dir / "colorbar.png", bbox_inches='tight')
                plt.close(cb_fig)
            else:
                print("Depth was not rendered or GT depth is not provided")

        if self.test_cfg.save_mesh and self.test_cfg.save_mesh_renders:
            for scale_res in [1, 2, 4]:
                mesh_rasterizer = MeshRasterizer(RasterizationSettings((h*scale_res, w*scale_res)), use_opengl=False)
                mesh_renderer = MeshRenderer(mesh_rasterizer)

                mesh_render_pkg = mesh_renderer(
                    convert_vertices_to_meshes(mesh), camera_poses, intrinsics, 
                    near, far, 
                    return_depth=True, return_normals=True, return_white_texture=True
                )

                mesh_depths = mesh_render_pkg["depth"]
                mesh_normals = mesh_render_pkg["normals"]
                mesh_whites = mesh_render_pkg["white_texture"]

                save_depth_dir = path / "images" / scene / f"mesh_depth_{scale_res}"
                os.makedirs(save_depth_dir, exist_ok=True)
                save_normal_dir = path / "images" / scene / f"mesh_normal_{scale_res}"
                os.makedirs(save_normal_dir, exist_ok=True)
                save_white_dir = path / "images" / scene / f"mesh_white_{scale_res}"
                os.makedirs(save_white_dir, exist_ok=True)

                num_images = len(mesh_depths)
                
                world_view_transforms = rearrange(camera_poses.inverse(), "b i j -> b j i")

                for i in range(num_images):
                    mesh_depth = mesh_depths[i].squeeze().detach().cpu() # (H, W)
                    mesh_depth_zero_mask = mesh_depth <= 0
                    depth_viz = viz_depth_tensor(
                        1.0 / mesh_depth.clamp(min=1e-6), return_numpy=True
                    )  # [H, W, 3]
                    Image.fromarray(depth_viz).save(save_depth_dir / f"{indices[i]:0>6}.png")
                    save_path = save_depth_dir / f"{indices[i]:0>6}.npy"
                    np.save(save_path, mesh_depth)

                    mesh_normal_raw = mesh_normals[i].squeeze().detach() # (H, W, 3)
                    mesh_normal = torch.nn.functional.normalize(mesh_normal_raw, p=2, dim=-1)
                    
                    mesh_normal_save = mesh_normal.clone().cpu()
                    mesh_normal_save[mesh_depth_zero_mask] = -1.0
                    mesh_normal_save = mesh_normal_save.permute(2, 0, 1)  # (3, H, W)
                    save_image((mesh_normal_save + 1) / 2.0, save_normal_dir / f"{indices[i]:0>6}.png")

                    mesh_white = mesh_whites[i].squeeze().detach() # (H, W, 3)
                    
                    mesh_normal_inv = -mesh_normal 
                    
                    R_wvt = world_view_transforms[i, :3, :3]
                    normal_view = torch.matmul(mesh_normal_inv, R_wvt)
                    normal_view = torch.nn.functional.normalize(normal_view, p=2, dim=-1)

                    light_dir = torch.tensor([0.3, 0.5, 1.0], device=normal_view.device)
                    light_dir = torch.nn.functional.normalize(light_dir, p=2, dim=0)

                    diffuse = torch.clamp(torch.matmul(normal_view, light_dir), min=0.0)

                    base_color = torch.tensor([0.85, 0.85, 0.85], device=normal_view.device) 
                    ambient_intensity = 0.2
                    diffuse_intensity = 0.8

                    lighting = ambient_intensity + (diffuse_intensity * diffuse)
                    shaded_color = base_color * lighting[..., None]

                    mask = mesh_white[..., 0:1] 
                    shaded_color = shaded_color * mask

                    shaded_img = (shaded_color.cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(shaded_img).save(save_white_dir / f"{indices[i]:0>6}.png")


        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])[:20]
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
                self.time_skip_steps_dict["extraction"] += 1

            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

            # compute depth metrics on predicted depths
            ctx_depth_gt = batch['context']["depth"] if "depth" in batch['context'] else None
            if pred_depths is not None and ctx_depth_gt is not None:
                if f"pred_abs_rel" not in self.test_step_outputs:
                    self.test_step_outputs[f"pred_abs_rel"] = []
                if f"pred_rmse" not in self.test_step_outputs:
                    self.test_step_outputs[f"pred_rmse"] = []
                if f"pred_rmse_log" not in self.test_step_outputs:
                    self.test_step_outputs[f"pred_rmse_log"] = []
                if f"pred_a1" not in self.test_step_outputs:
                    self.test_step_outputs[f"pred_a1"] = []

                pred_depths = pred_depths[0]  # [V, H, W]
                ctx_depth_gt = ctx_depth_gt[0]  # [V, H, W]

                ctx_near = batch["context"]["near"][...,
                                                None, None][0]  # [V, 1, 1]
                ctx_far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (ctx_depth_gt >= ctx_near) & (ctx_depth_gt <= ctx_far)

                all_metrics = compute_depth_errors(ctx_depth_gt[valid].detach().cpu().numpy(),
                                                   pred_depths[valid].detach().cpu().numpy())
                print("Predicted depth metrics:")
                print(all_metrics)

                self.test_step_outputs[f"pred_abs_rel"].append(
                    float(all_metrics[0]))
                self.test_step_outputs[f"pred_rmse"].append(float(all_metrics[2]))
                self.test_step_outputs[f"pred_rmse_log"].append(float(all_metrics[3]))
                self.test_step_outputs[f"pred_a1"].append(float(all_metrics[4]))
            
            rendered_depth = output.depth
            # compute depth metrics on predicted depths
            if rendered_depth is not None and depth_gt is not None:
                if f"rend_abs_rel" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_abs_rel"] = []
                if f"rend_rmse" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_rmse"] = []
                if f"rend_rmse_log" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_rmse_log"] = []
                if f"rend_a1" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_a1"] = []
                if f"rend_mae" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_mae"] = []
                if f"rend_medae" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_medae"] = []
                if f"rend_ncomp" not in self.test_step_outputs:
                    self.test_step_outputs[f"rend_ncomp"] = []

                all_metrics = compute_rend_depth_errors(depth_gt.unsqueeze(0),
                                                        rendered_depth,
                                                        near.unsqueeze(0), 
                                                        far.unsqueeze(0),
                                                        intrinsics.unsqueeze(0),
                                                        h, w)
                print("Rendered depth metrics:")
                print(all_metrics)

                self.test_step_outputs[f"rend_abs_rel"].append(
                    float(all_metrics[0]))
                self.test_step_outputs[f"rend_rmse"].append(float(all_metrics[2]))
                self.test_step_outputs[f"rend_rmse_log"].append(float(all_metrics[3]))
                self.test_step_outputs[f"rend_a1"].append(float(all_metrics[4]))
                self.test_step_outputs[f"rend_mae"].append(float(all_metrics[7]))
                self.test_step_outputs[f"rend_medae"].append(float(all_metrics[8]))
                self.test_step_outputs[f"rend_ncomp"].append(float(all_metrics[9]))

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()


    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, t, _, h, w = batch["target"]["image"].shape
        _, v, _, _, _ = batch["context"]["image"].shape
        assert b == 1

        pred_depths = None

        if self.test_cfg.inference_window_size is not None:
            assert self.test_cfg.inference_window_size > 0
            window = self.test_cfg.inference_window_size
            total_views = batch["context"]["image"].size(1)

            window_indices = sliding_window_indices(total_views, window, 0)

            all_gaussians = []
            all_states = []  # for global refinement

            # sliding window inference
            for indices in window_indices:
                start, end = indices
                curr_window_input = {
                    "extrinsics": batch["context"]["extrinsics"][:, start:end],
                    "intrinsics": batch["context"]["intrinsics"][:, start:end],
                    "image": batch["context"]["image"][:, start:end],
                    "near": batch["context"]["near"][:, start:end],
                    "far": batch["context"]["far"][:, start:end],
                    "index": batch["context"]["index"][:, start:end],
                }

                curr_gaussians = self.encoder(
                    curr_window_input,
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(curr_gaussians, dict):
                    pred_depths = curr_gaussians["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    if "condition_features" in curr_gaussians:
                        condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                    curr_gaussians = curr_gaussians["gaussians"]

                all_gaussians.append(curr_gaussians)
                all_states.append(condition_features)

            # merge all gaussians
            gaussians = Gaussians(
                torch.cat([g.means for g in all_gaussians], dim=1),
                torch.cat([g.covariances for g in all_gaussians], dim=1),
                torch.cat([g.harmonics for g in all_gaussians], dim=1),
                torch.cat([g.opacities for g in all_gaussians], dim=1),
                scales=torch.cat([g.scales for g in all_gaussians], dim=1),
                rotations=torch.cat([g.rotations for g in all_gaussians], dim=1),
                rotations_unnorm=torch.cat([g.rotations_unnorm for g in all_gaussians], dim=1),
            )

            # global refine after simply combining local window gaussians
            if self.encoder.cfg.num_refine > 0:
                # combine all condition features
                out = []
                is_ori_feature = True
                for i in range(len(all_states)):
                    curr = all_states[i]
                    if curr.dim() == 4:
                        # [BV, C, H, W]
                        curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                        is_ori_feature = True
                    elif curr.dim() == 2:
                        # [BVHW, C]
                        curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b,
                        h=h // self.encoder.cfg.latent_downsample,
                        w=w // self.encoder.cfg.latent_downsample,
                        )
                        is_ori_feature = False
                    else:
                        raise NotImplementedError

                    out.append(curr)

                if is_ori_feature:
                    concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                    concat = rearrange(concat, "b v c h w -> (b v) c h w")
                else:
                    concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                    concat = rearrange(concat, "b v h w c -> (b v) c h w")

                condition_features = concat

            gaussians_softmax = gaussians

        else:
            gaussians_softmax = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )

            if isinstance(gaussians_softmax, dict):
                pred_depths = gaussians_softmax["depths"]
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            if "condition_features" in gaussians_softmax:
                condition_features = gaussians_softmax["condition_features"]
            gaussians_softmax = gaussians_softmax["gaussians"]

        if self.train_cfg.mesh_nvs_weight != 0.:
            mesh_vertices = initialize_mesh_vertices(gaussians_softmax,
                            batch["context"]["extrinsics"], batch["context"]["intrinsics"],
                            batch["context"]["near"], batch["context"]["far"], 
                            (h, w), 'gggs', self.train_cfg.opacity_threshold, self.train_cfg.num_binary_steps, use_gpu_del=True,
                            remove_duplicates=self.train_cfg.remove_duplicates, vertex_corner_type=self.train_cfg.vertex_corner_type)
            is_empty = any(v.shape[0] == 0 for v in mesh_vertices.positions) or \
                    any(f.shape[0] == 0 for f in mesh_vertices.faces)
                    
            if is_empty:
                return None
            tgt_output_softmax = self.mesh_decoder.forward(
                mesh_vertices,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                sigma=self.train_cfg.triangle_sigma,
            )
            ctx_output_softmax = self.mesh_decoder.forward(
                mesh_vertices,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                sigma=self.train_cfg.triangle_sigma,
            )

        else:
            tgt_output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

            ctx_output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (h, w),
                depth_mode=None,
            )

        # refine
        if self.encoder.cfg.num_refine > 0:
            refine_output = self.encoder.forward_update(
                batch["context"],
                batch["target"],
                condition_features,
                gaussians_softmax,
                self.decoder,
                batch["context_remain"] if "context_remain" in batch.keys() else None,
            )

            render_output = refine_output['render']

            tgt_output_softmax = render_output[-1]

        rgb_softmax = tgt_output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(("val",), (rgb_softmax,)):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            # downsample image to depth resolution
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                input_images = F.interpolate(
                    batch["context"]["image"][0],
                    size=pred_depths.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)
            else:
                input_images = batch["context"]["image"][0]  # [N, 3, H, W]

            inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            # reshape when the number of input images is too large
            # otherwise the image will be too wide
            num_inputs = input_images.shape[0]
            width = input_images.shape[-1]
            if num_inputs > 8:
                rows = 4
                assert num_inputs % rows == 0
                stride = num_inputs // rows
                out = []
                for i in range(rows):
                    out.append(concat[:, :, width * stride * i : width * stride * (i + 1)])

                concat = torch.cat(out, dim=1)  # [3, H*2*R, W*N/R]

                # resize to half resolution to save space
                concat = F.interpolate(concat.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=True).squeeze(0)

            # viz gt depth
            if False:
                tmp_gt = batch["context"]["depth"][0]  # [V, H, W]
                tmp_gt[tmp_gt == 0] = 999999.
                tmp_inv_gt = 1. / tmp_gt
                tmp_concat = [tmp for tmp in tmp_inv_gt]
                tmp_concat = torch.cat(tmp_concat, dim=1)  # [H, W*N]
                tmp_concat = viz_depth_tensor(tmp_concat.cpu().detach())
                concat = torch.cat((concat, tmp_concat), dim=1)

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if self.train_cfg.viz_img_warp:
            context_imgs = batch["context"]["image"]  # [B, V, 3, H, W]
            
            context_intrinsics = batch["context"]["intrinsics"].clone()  # [B, V, 3, 3]
            context_intrinsics[:, :, 0] *= w
            context_intrinsics[:, :, 1] *= h
            context_extrinsics = batch["context"]["extrinsics"]  # [B, V, 4, 4]

            render_target_depths = tgt_output_softmax.depth # [B, T, H, W]
            render_context_depths = ctx_output_softmax.depth  # [B, V, H, W]

            target_imgs = batch["target"]["image"]  # [B, T, 3, H, W]
            target_intrinsics = batch["target"]["intrinsics"].clone()  # [B, T, 3, 3]

            target_intrinsics[:, :, 0] *= w
            target_intrinsics[:, :, 1] *= h
            target_extrinsics = batch["target"]["extrinsics"]  # [B, T, 4, 4]

            # relative pose from target to context
            # context_inv is world to context camera
            # target extrinsics is target camera to world
            context_inv = context_extrinsics.unsqueeze(1).inverse()  
            target2context = torch.matmul(context_inv, target_extrinsics.unsqueeze(2))

            intrinsics = target_intrinsics.unsqueeze(2).expand(-1, -1, v, -1, -1)  # [B, T, V, 3, 3]

            context_imgs = context_imgs.unsqueeze(1).expand(-1, t, -1, -1, -1, -1)  # [B, T, V, 3, H, W]
            render_target_depths = render_target_depths.unsqueeze(2).expand(-1, -1, v, -1, -1)  # [B, T, V, H, W]
            render_context_depths = render_context_depths.unsqueeze(1).expand(-1, t, -1, -1, -1)  # [B, T, V, H, W]

            # given the depth from target image and the pose from target to context, then unproject the depth to 3d coordinates
            # project those 3d coordinates to the context 2d coordinates, use these 2d coordinates to get the corresponding pixel
            # values, these should look then like the images from the target camera poses. kind of like tracing the rays twice to the
            # 3d scene (as devised by the target depth map) and then to the context image
            warped_imgs, warp_valid_masks = warp_with_depth_pose(
                rearrange(context_imgs, "b t v c h w -> (b t v) c h w"),
                rearrange(intrinsics, "b t v x y -> (b t v) x y"),
                rearrange(target2context, "b t v x y -> (b t v) x y"),
                rearrange(render_target_depths, "b t v h w -> (b t v) h w"),
                depth_tgt=rearrange(render_context_depths.detach(), "b t v h w -> (b t v) h w"),
                return_mask=True, 
            )  # [BTV, 3, H, W], [BTV, H, W]
            warped_imgs *= warp_valid_masks.unsqueeze(1).float()

            # batch == 1 so can be ignored now
            # target_imgs is [B, T, 3, H, W]
            # layout is T rows and V+1 cols with the +1 being the gt target image
            # logged images should be of shape 3, H, W so shapes should be
                # target_imgs_col: [3, H*T, W]
                # all_others: [3, H*T, W*V]
            context_imgs_row = rearrange(batch["context"]["image"][0], "v c h w -> c h (v w)") 
            target_imgs_col = rearrange(target_imgs[0], "t c h w -> c (t h) w")
            warped_imgs_grid = rearrange(warped_imgs, "(t v) c h w -> c (t h) (v w)", t=t, v=v)
            grid = torch.cat((target_imgs_col, warped_imgs_grid), dim=2) # [3, H*T, W*(V+1)]

            dark_rectangle = torch.zeros((3, h, w), device=context_imgs_row.device, dtype=torch.float)
            context_imgs_row = torch.cat((dark_rectangle, context_imgs_row), dim=2) # [3, H, W*(V+1)]

            grid = torch.cat((context_imgs_row, grid), dim=1)
            grid = (grid * 255).cpu().detach()

            self.logger.log_image(
                "img_warp",
                [grid],
                step=self.global_step,
                caption=batch["scene"],
            )


        # viz rendered normal, normal depth in camera coordinates, and the error
        if self.train_cfg.viz_geo_renders:
            rendered_img_batch = tgt_output_softmax.color[0] # [V, 3, H, W]
            rendered_depth_batch = tgt_output_softmax.depth[0] # [V, H, W]
            gt_depth_batch = batch["target"]["depth"][0] # [V, H, W]
            rendered_normal_batch = tgt_output_softmax.normal[0] # [V, 3, H, W]
            gt_normal_batch = batch["target"]["normal"][0] # [V, 3, H, W]
            intrinsics_batch = batch["target"]["intrinsics"][0] # [V, 3, 3]

            rendered_imgs, rendered_normals, depth_normals, rendered_depths, gt_depths, gt_normals = [], [], [], [], [], []
            xy_ray, _ = sample_image_grid((h, w), intrinsics_batch.device)
            for rendered_img, rendered_depth, intrinsics, rendered_normal, gt_normal, gt_depth in zip(rendered_img_batch, rendered_depth_batch, intrinsics_batch, rendered_normal_batch, gt_normal_batch, gt_depth_batch):
                # [H, W, 3]
                directions = get_camera_rays(xy_ray, intrinsics)
                point_cloud = directions * rendered_depth[..., None]

                # [1, 1, H, W, 3] as we are using the batch-views function
                point_cloud = point_cloud.unsqueeze(0).unsqueeze(0)
                # [1, 1, H, W, 3]
                depth_normal = torch.zeros_like(point_cloud)
                depth_normal_nomargins = pcd_to_normal(point_cloud)
                depth_normal_mask = get_valid_normal_mask(point_cloud)
                depth_normal_nomargins *= depth_normal_mask[..., None]
                depth_normal[:, :, 1:-1, 1:-1, :] = depth_normal_nomargins
                depth_normal = depth_normal.squeeze(0).squeeze(0) # [H, W, 3]
                depth_normal = depth_normal.permute(2, 0, 1) # [3, H, W]

                rendered_imgs.append(rendered_img)
                rendered_normals.append((rendered_normal + 1) / 2.0)
                gt_normals.append((gt_normal + 1) / 2.0)
                depth_normals.append((depth_normal + 1) / 2.0)

                inverse_rendered_depth = 1.0 / rendered_depth # [H, W]
                depth_viz = viz_depth_tensor(inverse_rendered_depth.cpu().detach())
                rendered_depths.append(depth_viz)

                inverse_gt_depth = 1.0 / gt_depth # [H, W]
                gt_depth_viz = viz_depth_tensor(inverse_gt_depth.cpu().detach())
                gt_depths.append(gt_depth_viz)

            rendered_imgs = torch.cat(rendered_imgs, dim=-1) * 255 # [3, H, W*N]
            rendered_normals = torch.cat(rendered_normals, dim=-1) * 255 # [3, H, W*N]
            depth_normals = torch.cat(depth_normals, dim=-1) * 255 # [3, H, W*N]
            gt_normals = torch.cat(gt_normals, dim=-1) * 255 # [3, H, W*N]
            rendered_depths = torch.cat(rendered_depths, dim=-1) # [3, H, W*N]
            gt_depths = torch.cat(gt_depths, dim=-1) # [3, H, W*N]

            # also concat gt images
            input_images = batch["target"]["image"][0]  # [N, 3, H, W]
            gt_images = [img for img in input_images]
            gt_images = torch.cat(gt_images, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (gt_images.cpu().detach(), rendered_imgs.cpu().detach(), 
                 gt_normals.cpu().detach(), rendered_normals.cpu().detach(), depth_normals.cpu().detach(),
                 gt_depths, rendered_depths), dim=1
            )  # [3, H*7, W*N]

            self.logger.log_image(
                "geo_renders",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if batch["context"]["image"][0].shape[0] > 16:
            # when the number of input images is too large
            # subsample input images to save space
            viz_input = batch["context"]["image"][0][::4]
            tag = "Context (1/4)"
        elif batch["context"]["image"][0].shape[0] > 8:
            # when the number of input images is too large
            # subsample input images to save space
            viz_input = batch["context"]["image"][0][::2]
            tag = "Context (1/2)"
        else:
            viz_input = batch["context"]["image"][0]
            tag = "Context"

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*viz_input), tag),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Prediction)"),
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        if not self.train_cfg.no_log_projections:
            # Render projections and construct projection image.
            projections = hcat(
                *render_projections(
                    gaussians_softmax,
                    256,
                    extra_label="(Prediction)",
                )[0]
            )
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        if not self.train_cfg.no_viz_video:
            self.render_video_interpolation(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        ctx_depth_gt = None
        tgt_depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        for score_tag in ("tgt_psnr", "tgt_ssim", "tgt_lpips",
                          "ctx_psnr", "ctx_ssim", "ctx_lpips"):
            scores_dict[score_tag] = {}
            for method_tag in ("deterministic", "probabilistic"):
                scores_dict[score_tag][method_tag] = []

        # evaluate input view depth
        if self.train_cfg.eval_depth:
            for score_tag in ("pred_abs_rel", "pred_rmse", "pred_rmse_log", "pred_a1",
                              "tgt_rend_abs_rel", "tgt_rend_rmse", "tgt_rend_rmse_log", 
                              "tgt_rend_a1", "tgt_rend_mae", "tgt_rend_medae", "tgt_rend_ncomp",
                              "ctx_rend_abs_rel", "ctx_rend_rmse", "ctx_rend_rmse_log", 
                              "ctx_rend_a1", "ctx_rend_mae", "ctx_rend_medae", "ctx_rend_ncomp"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                if self.test_cfg.inference_window_size is not None:
                    assert self.test_cfg.inference_window_size > 0
                    window = self.test_cfg.inference_window_size
                    total_views = batch["context"]["image"].size(1)

                    window_indices = sliding_window_indices(total_views, window, 0)

                    all_gaussians = []
                    all_states = []  # for global refinement

                    # sliding window inference
                    for indices in window_indices:
                        start, end = indices
                        curr_window_input = {
                            "extrinsics": batch["context"]["extrinsics"][:, start:end],
                            "intrinsics": batch["context"]["intrinsics"][:, start:end],
                            "image": batch["context"]["image"][:, start:end],
                            "near": batch["context"]["near"][:, start:end],
                            "far": batch["context"]["far"][:, start:end],
                            "index": batch["context"]["index"][:, start:end],
                        }

                        curr_gaussians = self.encoder(
                            curr_window_input,
                            self.global_step,
                            deterministic=False,
                        )

                        if isinstance(curr_gaussians, dict):
                            pred_depths = curr_gaussians["depths"]
                            if "depth" in batch["context"]:
                                ctx_depth_gt = batch["context"]["depth"]
                                tgt_depth_gt = batch["target"]["depth"]
                            if "condition_features" in curr_gaussians:
                                condition_features = curr_gaussians["condition_features"]  # [BV, C, H, W]
                            else:
                                condition_features = None
                            curr_gaussians = curr_gaussians["gaussians"]

                        all_gaussians.append(curr_gaussians)
                        all_states.append(condition_features)

                    # merge all gaussians
                    # ['means', 'covariances', 'harmonics', 'opacities']
                    gaussians = Gaussians(
                        torch.cat([g.means for g in all_gaussians], dim=1),
                        torch.cat([g.covariances for g in all_gaussians], dim=1),
                        torch.cat([g.harmonics for g in all_gaussians], dim=1),
                        torch.cat([g.opacities for g in all_gaussians], dim=1),
                        scales=torch.cat([g.scales for g in all_gaussians], dim=1),
                        rotations=torch.cat([g.rotations for g in all_gaussians], dim=1),
                        rotations_unnorm=torch.cat([g.rotations_unnorm for g in all_gaussians], dim=1),
                    )

                    gaussians_probabilistic = gaussians

                    # global refine after simply combining local window gaussians
                    if self.encoder.cfg.num_refine > 0:
                        # combine all condition features
                        out = []
                        is_ori_feature = False
                        for i in range(len(all_states)):
                            curr = all_states[i]
                            if curr.dim() == 4:  
                                # [BV, C, H, W]
                                curr = rearrange(curr, "(b v) c h w -> b v c h w", b=b)
                                is_ori_feature = True
                            elif curr.dim() == 2:
                                # [BVHW, C]
                                curr = rearrange(curr, "(b v h w) c -> b v h w c", b=b, 
                                h=h // self.encoder.cfg.latent_downsample,
                                w=w // self.encoder.cfg.latent_downsample,
                                )
                                is_ori_feature = False
                            else:
                                raise NotImplementedError

                            out.append(curr)

                        # concat
                        if is_ori_feature:
                            concat = torch.cat(out, dim=1)  # [B, V*K, C, H, W]
                            concat = rearrange(concat, "b v c h w -> (b v) c h w")
                        else:
                            concat = torch.cat(out, dim=1)  # [B, V*K, H, W, C]
                            concat = rearrange(concat, "b v h w c -> (b v) c h w")

                        condition_features = concat

                else:
                    gaussians_probabilistic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=False,
                    )

                    if isinstance(gaussians_probabilistic, dict):
                        pred_depths = gaussians_probabilistic["depths"]
                        if "depth" in batch["context"]:
                            ctx_depth_gt = batch["context"]["depth"]
                            tgt_depth_gt = batch["target"]["depth"]
                        if "condition_features" in gaussians_probabilistic:
                            condition_features = gaussians_probabilistic["condition_features"]
                        gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            with self.benchmarker.time("decoder", num_calls=v):
                tgt_output_probabilistic = self.decoder.forward(
                    gaussians_probabilistic,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )

                ctx_output_probabilistic = self.decoder.forward(
                    gaussians_probabilistic,
                    batch["context"]["extrinsics"],
                    batch["context"]["intrinsics"],
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (h, w),
                    depth_mode=None,
                )

                # refine
                if self.encoder.cfg.num_refine > 0:
                    refine_output = self.encoder.forward_update(
                        batch["context"],
                        batch["target"],
                        condition_features,
                        gaussians_probabilistic,
                        self.decoder,
                        batch["context_remain"] if "context_remain" in batch.keys() else None,
                    )

                    render_output = refine_output['render']

                    tgt_output_probabilistic = render_output[-1]

            rgbs = [tgt_output_probabilistic.color[0]]
            tags = ["probabilistic"]

            if self.train_cfg.eval_deterministic:
                gaussians_deterministic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=True,
                )
                output_deterministic = self.decoder.forward(
                    gaussians_deterministic,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
                rgbs.append(output_deterministic.color[0])
                tags.append("deterministic")

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(tags, rgbs):
                scores_dict["tgt_psnr"][tag].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                scores_dict["tgt_lpips"][tag].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
                scores_dict["tgt_ssim"][tag].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )

            # Compute validation metrics.
            rgb_gt = batch["context"]["image"][0]
            rgbs = [ctx_output_probabilistic.color[0]]
            for tag, rgb in zip(tags, rgbs):
                scores_dict["ctx_psnr"][tag].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                scores_dict["ctx_lpips"][tag].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
                scores_dict["ctx_ssim"][tag].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )

            # compute depth metrics on predicted depth map
            if pred_depths is not None and ctx_depth_gt is not None and ctx_depth_gt.max() > 0 and self.train_cfg.eval_depth:
                pred_depths_views = pred_depths[0]  # [V, H, W]

                # gaussian downsample
                if pred_depths_views.shape[1:] != batch["context"]["image"].shape[-2:]:
                    pred_depths_views = F.interpolate(
                        pred_depths_views.unsqueeze(1),
                        size=batch["context"]["image"].shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(1)

                ctx_depth_gt_views = ctx_depth_gt[0]  # [V, H, W]

                near = batch["context"]["near"][...,
                                                None, None][0]  # [V, 1, 1]
                far = batch["context"]["far"][..., None, None][0]  # [V, 1, 1]

                valid = (ctx_depth_gt_views >= near) & (ctx_depth_gt_views <= far)

                all_metrics = compute_depth_errors(ctx_depth_gt_views[valid].detach().cpu().numpy(),
                                                   pred_depths_views[valid].detach().cpu().numpy())
                scores_dict["pred_abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["pred_rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["pred_rmse_log"]["probabilistic"].append(all_metrics[3])
                scores_dict["pred_a1"]["probabilistic"].append(all_metrics[4])

            tgt_rendered_depth = tgt_output_probabilistic.depth
            # compute depth metrics on rendered depth map
            if tgt_rendered_depth is not None and tgt_depth_gt is not None and tgt_depth_gt.max() > 0 and self.train_cfg.eval_depth:             
                near = batch["target"]["near"] # [B, V]
                far = batch["target"]["far"]   # [B, V]
                intrinsics = batch["target"]["intrinsics"]   # [B, V, 3, 3]

                all_metrics = compute_rend_depth_errors(tgt_depth_gt,
                                                        tgt_rendered_depth,
                                                        near, far,
                                                        intrinsics,
                                                        h, w)
                
                scores_dict["tgt_rend_abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["tgt_rend_rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["tgt_rend_rmse_log"]["probabilistic"].append(all_metrics[3])
                scores_dict["tgt_rend_a1"]["probabilistic"].append(all_metrics[4])
                scores_dict["tgt_rend_mae"]["probabilistic"].append(all_metrics[7])
                scores_dict["tgt_rend_medae"]["probabilistic"].append(all_metrics[8])
                scores_dict["tgt_rend_ncomp"]["probabilistic"].append(all_metrics[9])

            ctx_rendered_depth = ctx_output_probabilistic.depth # target
            # compute depth metrics on rendered depth map
            if ctx_rendered_depth is not None and ctx_depth_gt is not None and ctx_depth_gt.max() > 0 and self.train_cfg.eval_depth:             
                near = batch["context"]["near"] # [B, V]
                far = batch["context"]["far"]   # [B, V]
                intrinsics = batch["context"]["intrinsics"]   # [B, V, 3, 3]

                all_metrics = compute_rend_depth_errors(ctx_depth_gt,
                                                        ctx_rendered_depth,
                                                        near, far,
                                                        intrinsics,
                                                        h, w)
                
                scores_dict["ctx_rend_abs_rel"]["probabilistic"].append(all_metrics[0])
                scores_dict["ctx_rend_rmse"]["probabilistic"].append(all_metrics[2])
                scores_dict["ctx_rend_rmse_log"]["probabilistic"].append(all_metrics[3])
                scores_dict["ctx_rend_a1"]["probabilistic"].append(all_metrics[4])
                scores_dict["ctx_rend_mae"]["probabilistic"].append(all_metrics[7])
                scores_dict["ctx_rend_medae"]["probabilistic"].append(all_metrics[8])
                scores_dict["ctx_rend_ncomp"]["probabilistic"].append(all_metrics[9])

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        if self.train_cfg.no_log_video:
            return
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_depth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "depth_predictor" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_depth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_depth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        elif self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_monodepth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": pretrained_params,
                            "lr": self.optimizer_cfg.lr_monodepth,
                        },
                        {"params": new_params, "lr": self.optimizer_cfg.lr},
                    ],
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            if self.optimizer_cfg.adamw_8bit:
                optimizer = AdamW8bit(
                    self.parameters(),
                    lr=self.optimizer_cfg.lr,
                    weight_decay=self.optimizer_cfg.weight_decay,
                )
            else:
                optimizer = optim.AdamW(
                    self.parameters(),
                    lr=self.optimizer_cfg.lr,
                    weight_decay=self.optimizer_cfg.weight_decay,
                )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=self.optimizer_cfg.warm_up_ratio,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    ref: https://github.com/nianticlabs/monodepth2/blob/master/layers.py#L218
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def sliding_window_indices(N, x, y):
    indices = []
    start = 0
    while start + x < N:  # Ensure the last window is not processed here
        end = min(start + x, N)
        indices.append([start, end])
        start += (x - y)  # Move the start by the window size minus overlap
    
    # Append the last window [N-x, N]
    indices.append([N - x, N])
    
    return indices