from os.path import join, exists
from os import mkdir
from glob import glob


def CheckFolder(foldername: str) -> None:
    if exists(foldername) is False:
        mkdir(foldername)


dataset_folder = "datasets"
outputs_folder = "outputs"
third_party_folder = "third-party"
images_dataset_folder = "images"
models_folder = "models"
upscalers_folder = join("third-party", "upscaler_models")
temp_output_folder = "temp_output"
input_folder = "inputs"


def get_multiple_extension_path(folder: str, extensions: list[str]) -> list[str]:
    paths = []

    for extension in extensions:
        paths.extend(glob(join(folder, f"*{extension}")))

    return paths


def get_dataset_paths() -> list[str]:
    return glob(join(dataset_folder, "*.pickle"))


def get_output_image_paths() -> list[str]:
    return glob(join(outputs_folder, "*"))


def get_dataset_images_paths() -> list[str]:
    extensions = [".png", ".ppm", ".jpeg", ".jpg", ".webp"]

    return get_multiple_extension_path(images_dataset_folder, extensions)


def get_model_script_paths() -> list[str]:
    return glob(join(models_folder, "*.py"))


def get_model_state_dict_paths() -> list[str]:
    extensions = [".pt", ".pth"]

    return get_multiple_extension_path(models_folder, extensions)


def get_upscalers_paths() -> list[str]:
    return glob(join(upscalers_folder, "*"))


def get_temp_output_images_paths() -> list[str]:
    return glob(join(temp_output_folder, "*"))


def get_test_input_paths() -> list[str]:
    extensions = [".png", ".ppm", ".jpeg", ".jpg", ".webp"]

    return get_multiple_extension_path(input_folder, extensions)
