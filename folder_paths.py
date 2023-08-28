from os.path import join, exists
from os import mkdir


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
