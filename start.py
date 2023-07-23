from VenvManager import VenvManager

from os.path import exists, join
from os import mkdir

def CheckFolder(foldername : str) -> None:
    if exists(foldername) is False:
        mkdir(foldername)

if __name__ == "__main__":
    venv_manager = VenvManager()
    realesrgan_repokey = "realesrgan"

    CheckFolder("datasets")
    CheckFolder("outputs")
    CheckFolder("third-party")

    venv_manager.InstallWRequirements()

    if exists(join("third-party", "Real-ESRGAN")) is False:
        venv_manager.CloneRepository("https://github.com/xinntao/Real-ESRGAN.git", realesrgan_repokey, "third-party")
        venv_manager.InstallRequirementsFromRepository(realesrgan_repokey)
        venv_manager.RunScriptInsideRepository(realesrgan_repokey, "setup.py develop")

    venv_manager.RunScript("main")