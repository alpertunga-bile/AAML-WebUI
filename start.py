from VenvManager import VenvManager

from folder_paths import *


if __name__ == "__main__":
    venv_manager = VenvManager()

    CheckFolder(dataset_folder)
    CheckFolder(input_folder)
    CheckFolder(outputs_folder)
    CheckFolder(third_party_folder)
    CheckFolder(images_dataset_folder)
    CheckFolder(models_folder)
    CheckFolder(upscalers_folder)

    venv_manager.InstallWRequirements()

    venv_manager.RunScript("main")
