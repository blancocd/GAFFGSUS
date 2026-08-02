from ...dataset import DatasetCfg
from .decoder import Decoder
# from .mip_decoder_splatting_cuda import MipDecoderSplattingCUDA, MipDecoderSplattingCUDACfg
from .gsplat_decoder_splatting_cuda import GSplatDecoderSplattingCUDACfg, GSplatDecoderSplattingCUDA
from .gof_decoder_splatting_cuda import GOFDecoderSplattingCUDA, GOFDecoderSplattingCUDACfg
from .gggs_decoder_splatting_cuda import GGGSDecoderSplattingCUDA, GGGSDecoderSplattingCUDACfg
from .triangle_decoder_splatting_cuda import TriangleDecoderSplattingCUDA, TriangleDecoderSplattingCUDACfg


DECODERS = {
    # "mip": MipDecoderSplattingCUDA,
    "gsplat": GSplatDecoderSplattingCUDA,
    "gof": GOFDecoderSplattingCUDA,
    "gggs": GGGSDecoderSplattingCUDA,
    "triangle_splatting": TriangleDecoderSplattingCUDA,
}

# DecoderCfg = MipDecoderSplattingCUDACfg | GSplatDecoderSplattingCUDACfg | GGGSDecoderSplattingCUDACfg | GOFDecoderSplattingCUDACfg | TriangleDecoderSplattingCUDACfg
DecoderCfg = GSplatDecoderSplattingCUDACfg | GGGSDecoderSplattingCUDACfg | GOFDecoderSplattingCUDACfg | TriangleDecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)

def get_mesh_decoder(dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS['triangle_splatting'](TriangleDecoderSplattingCUDACfg('triangle_splatting', None), dataset_cfg)
