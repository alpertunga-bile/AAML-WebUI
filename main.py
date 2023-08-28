from gradio import (
    Blocks,
    Button,
    Code,
    Dropdown,
    File,
    Gallery,
    Image,
    Label,
    Plot,
    Row,
    Tab,
    Text,
    Slider,
)
from gradio.themes import Monochrome
from os import listdir
from os.path import isfile

with Blocks(title="AAML WebUI", theme=Monochrome()) as application:
    with Tab("Train"):
        dataset_field = File(
            label="Choose Dataset",
            type="file",
            file_types=[".pickle"],
            interactive=True,
        )
        model_dropdown = Dropdown(
            choices=[f for f in listdir("models/") if isfile(f)],
            label="Models",
            interactive=True,
        )
        with Row():
            epoch_slider = Slider(
                1, 10000, value=20, step=1, label="Epochs", interactive=True
            )
            batch_slider = Slider(1, 4096, step=1, label="Batch Size", interactive=True)
        with Row():
            train_plot = Plot(label="Train Plot")
            test_plot = Plot(label="Test Plot")
        train_button = Button(value="Train")
    with Tab("Test"):
        with Row():
            test_image = Image(image_mode="RGBA", source="upload", type="pil")
            output_image = Gallery(label="Just DL Model Output Image")
            upscale_output_image = Gallery(label="With Upscale Method Output Image")
        with Row():
            tile_size_slider = Slider(
                0, 1024, value=192, step=1, label="Tile Size", interactive=True
            )
            upscale_model_dropdown = Dropdown(
                choices=[
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "realesr-general-x4v3",
                ],
                value="RealESRGAN_x4plus",
                label="Models",
                interactive=True,
            )
        test_button = Button(value="Run")
    with Tab("Model"):
        custom_model_code = Code(
            value="print('Hello World')", language="python", lines=25, interactive=True
        )
        save_custom_model = Button("Save Model")
    with Tab("Dataset"):
        dataset_file = File(
            label="Choose Dataset",
            type="file",
            file_types=[".pickle"],
            interactive=True,
        )
        new_dataset_text = Text(
            value="None",
            lines=1,
            placeholder="Specify if not choosing existed dataset file",
            label="Choose Dataset Name",
            interactive=True,
        )
        with Row():
            kernel_size_slider = Slider(
                minimum=3,
                maximum=21,
                value=3,
                step=2,
                label="Kernel One Dimension",
                interactive=True,
            )
            stride_slider = Slider(
                minimum=1, maximum=21, value=1, step=1, label="Stride", interactive=True
            )
            max_row_slider = Slider(
                minimum=20000000,
                maximum=100000000,
                step=2500000,
                label="Max Row For Dataset",
                interactive=True,
            )
        generate_button = Button("Generate")

if __name__ == "__main__":
    application.queue(concurrency_count=4).launch()
