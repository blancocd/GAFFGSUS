from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Vertices
from .triangle_cuda_splatting import render, DepthRenderingMode
from .decoder import Decoder, DecoderOutput


@dataclass
class TriangleDecoderSplattingCUDACfg:
    name: Literal["triangle_splatting"]
    alpha_normalization_threshold: float | None


class TriangleDecoderSplattingCUDA(Decoder[TriangleDecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: TriangleDecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    # in mesh splatting paper sigma varies from 1 to 0.0001
    def forward(
        self,
        vertices: Vertices,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        sigma: float,
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        
        triangles_indices = []
        positions_list = []
        harmonics_list = []
        weights_list = []
        # this should copy pointers and not create extra gpu mem
        for i in range(b):
            triangles_indices.extend([vertices.faces[i]] * v)
            positions_list.extend([vertices.positions[i]] * v)
            harmonics_list.extend([vertices.harmonics[i]] * v)
            weights_list.extend([vertices.weights[i]] * v)

        # [B, 3, H, W], [B, 3, H, W], [B, H, W], [B, H, W]
        color, normals, expected_depths, median_depths, alphas = render(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            positions_list,
            triangles_indices,
            harmonics_list,
            weights_list,
            sigma=sigma,
            alpha_normalization_threshold=self.cfg.alpha_normalization_threshold
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        normals = rearrange(normals, "(b v) c h w -> b v c h w", b=b, v=v)
        expected_depths = rearrange(expected_depths, "(b v) h w -> b v h w", b=b, v=v)
        median_depths = rearrange(median_depths, "(b v) h w -> b v h w", b=b, v=v)
        alphas = rearrange(alphas, "(b v) h w -> b v h w", b=b, v=v)

        return DecoderOutput(
            color,
            depth=median_depths,
            normal=normals,
            accumulated_alpha=alphas
        )
