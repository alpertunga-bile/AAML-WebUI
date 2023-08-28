from upscaler_helpers import get_upscaler, get_face_enhancer

from folder_paths import temp_output_folder, outputs_folder
from glob import glob
from os.path import join, splitext, basename
from cv2 import imread, IMREAD_UNCHANGED, imwrite
from tqdm import tqdm


def upscale(
    model_name: str,
    scale_factor: float,
    tile_size: int,
    use_fp32: bool,
    face_enhance: bool,
):
    upscaler = get_upscaler(model_name, tile_size, use_fp32)

    if face_enhance:
        face_enhancer = get_face_enhancer(upscaler, scale_factor)

    image_paths = glob(join(temp_output_folder, "*"))

    for path in tqdm(image_paths, desc="Upscaling"):
        name, extension = splitext(basename(path))

        image = imread(path, IMREAD_UNCHANGED)

    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = upscaler.enhance(image, outscale=scale_factor)
    except RuntimeError as error:
        print("Use smaller tile size")
    else:
        save_path = join(outputs_folder, f"{name}.{extension}")
        imwrite(save_path, output)
