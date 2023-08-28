from VenvManager import VenvManager

from os.path import exists
from os import mkdir


def CheckFolder(foldername: str) -> None:
    if exists(foldername) is False:
        mkdir(foldername)


if __name__ == "__main__":
    venv_manager = VenvManager()
    realesrgan_repokey = "realesrgan"

    CheckFolder("datasets")
    CheckFolder("outputs")
    CheckFolder("third-party")
    CheckFolder("images")
    CheckFolder("models")
    CheckFolder("third-party")

    venv_manager.InstallWRequirements()

    venv_manager.RunScript("main")
