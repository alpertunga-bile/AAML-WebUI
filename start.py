from VenvManager import VenvManager

from folder_paths import *


if __name__ == "__main__":
    venv_manager = VenvManager()

    CheckFolder(dataset_folder)  # store the datasets
    CheckFolder(input_folder)  # test inputs
    CheckFolder(outputs_folder)  # test outputs
    CheckFolder(third_party_folder)  # storing third-party models
    CheckFolder(images_dataset_folder)  # images for the dataset
    CheckFolder(
        models_folder
    )  # deep learning models state dicts and scripts to train and test
    CheckFolder(upscalers_folder)  # Real-Esrgan models

    venv_manager.InstallWRequirements()

    venv_manager.RunScript("main")
