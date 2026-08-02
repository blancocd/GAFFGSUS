from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F

from ...dataset import DatasetCfg
from ..types import Gaussians
from .gggs_cuda_splatting import render_cuda
from .decoder import Decoder, DecoderOutput, DepthRenderingMode


@dataclass
class GGGSDecoderSplattingCUDACfg:
    name: Literal["gggs"]
    scale_invariant: bool


class GGGSDecoderSplattingCUDA(Decoder[GGGSDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: GGGSDecoderSplattingCUDACfg,
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
        color, radii, median_depth, alpha, normal = render_cuda(
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
        # color.shape,        median_depth.shape, alpha.shape,    normal.shape)
        # [B*V, 3, H, W]      [B*V, 1, H, W]      [B*V, 1, H, W]  [B*V, 3, H, W]

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        median_depth = rearrange(median_depth, "(b v) 1 h w -> b v h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) 1 h w -> b v h w", b=b, v=v)
        normal = rearrange(normal, "(b v) c h w -> b v c h w", b=b, v=v)
        normal = torch.nn.functional.normalize(normal, dim=2)

        return DecoderOutput(
            color=color,                
            depth=median_depth,
            accumulated_alpha=alpha,
            normal=normal,
            radii=radii if return_radii else None
        )
