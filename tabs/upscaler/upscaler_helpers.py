from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from typing import Union

from realesrgan import RealESRGANer
from gfpgan import GFPGANer

from folder_paths import upscalers_folder
from os.path import join, exists


def get_model_path(model_name: str, url: str) -> str:
    model_path = join(upscalers_folder, f"{model_name}.pth")

    if exists(model_path) is False:
        model_path = load_file_from_url(
            url=url, model_dir=upscalers_folder, progress=True, file_name=None
        )

    return model_path


def get_face_enhancer(upsampler: RealESRGANer, scale_factor: float) -> GFPGANer:
    model_path = join(upscalers_folder, "GFPGANv1.3.pth")

    if exists(model_path) is False:
        model_path = load_file_from_url(
            url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            model_dir=upscalers_folder,
            progress=True,
            file_name=None,
        )

    face_enhancer = GFPGANer(
        model_path=model_path,
        upscale=scale_factor,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler,
    )

    return face_enhancer


def get_upscaler(model_name: str, tile_size: int, use_fp32: bool) -> RealESRGANer:
    model, netscale, url = get_model_infos(model_name)
    filepath = get_model_path(model_name, url)

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=filepath,
        model=model,
        tile=tile_size,
        pre_pad=0,
        half=not use_fp32,
    )

    return upsampler


def get_model_infos(model_name: str) -> Union[RRDBNet, int, str]:
    if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"

    elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"

    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

    elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"

    return [model, netscale, file_url]
