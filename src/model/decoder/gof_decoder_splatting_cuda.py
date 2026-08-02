from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ...dataset import DatasetCfg
from ..types import Gaussians
from .gof_cuda_splatting import DepthRenderingMode, render_depth_cuda, render_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class GOFDecoderSplattingCUDACfg:
    name: Literal["gof"]
    scale_invariant: bool

class GOFDecoderSplattingCUDA(Decoder[GOFDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GOFDecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        return_radii: bool = False
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        render_pkgs = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v) if gaussians.covariances is not None else None,
            repeat(gaussians.scales, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.rotations_world, "b g xyzw -> (b v) g xyzw", v=v) if gaussians.rotations_world is not None else None,
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=self.cfg.scale_invariant,
            return_radii=return_radii,
        )

        if return_radii:
                render_pkgs, radii = color
        color = rearrange(render_pkgs[:, :3], "(b v) c h w -> b v c h w", b=b, v=v)
        # Normals were normalized and then blended as in the VRE with alpha and transmittance
        # So we better make sure it is a unit vector
        # in GOF code this normal is in camera coordinates, here too since the extrinsics are refering to world coordinates too
        normal = rearrange(F.normalize(render_pkgs[:, 3:6], p=2, dim=1), "(b v) c h w -> b v c h w", b=b, v=v)
        # from this depth map, one can also get an estimated normal map, we do need a camera viewpoint to unproject
        # after unprojecting this would be in world coordinates
        depth = rearrange(render_pkgs[:, 6], "(b v) h w -> b v h w", b=b, v=v)
        # i imagine we could give identity as extrinsics and corresponding intrinsics 
        # and then compare the normals with camera coordinates too, may be more stable

        accumulated_alpha = rearrange(render_pkgs[:, 7], "(b v) h w -> b v h w", b=b, v=v)
        # distortion map will be used for depth distortion loss
        distortion_map = rearrange(render_pkgs[:, 8], "(b v) h w -> b v h w", b=b, v=v)

        return DecoderOutput(
            color=color,
            depth=depth,
            normal=normal,
            distortion_map=distortion_map,
            accumulated_alpha=accumulated_alpha,
            radii=radii if return_radii else None
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)
