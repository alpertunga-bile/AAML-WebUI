from gradio import (
    Blocks,
    Button,
    Checkbox,
    Code,
    Dropdown,
    File,
    Gallery,
    Plot,
    Row,
    Tab,
    Text,
    Slider,
)
from gradio.themes import Monochrome
from folder_paths import (
    get_model_script_paths,
    get_model_state_dict_paths,
    get_test_input_paths,
)

with Blocks(title="AAML WebUI", theme=Monochrome()) as application:
    with Tab("Train"):
        dataset_field = File(
            label="Choose Dataset",
            type="file",
            file_types=[".pickle"],
            interactive=True,
        )
        with Row():
            train_model_script_dropdown = Dropdown(
                choices=get_model_script_paths(),
                label="Model Script",
                interactive=True,
            )
            train_model_state_dict_dropdown = Dropdown(
                choices=get_model_state_dict_paths(),
                label="Model State Dicts",
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
            test_image = Gallery(
                value=get_test_input_paths(),
                label="Used Input Images",
            )
            output_image = Gallery(label="Just DL Model Output Image")
            upscale_output_image = Gallery(label="With Upscale Method Output Image")
        with Row():
            test_model_script_dropdown = Dropdown(
                choices=get_model_script_paths(),
                label="Model Script",
                interactive=True,
            )
            test_model_state_dict_dropdown = Dropdown(
                choices=get_model_state_dict_paths(),
                label="Model State Dicts",
                interactive=True,
            )
        with Row():
            upscale_model_dropdown = Dropdown(
                choices=[
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "RealESRGAN_x4plus_anime_6B",
                    "RealESRGAN_x2plus",
                    "realesr-general-wdn-x4v3",
                ],
                value="RealESRGAN_x4plus",
                label="Models",
                interactive=True,
            )
            tile_size_slider = Slider(
                0, 1024, value=192, step=1, label="Tile Size", interactive=True
            )
            scale_factor_slider = Slider(
                1, 4, value=2, step=0.1, label="Scale", interactive=True
            )
            fp32_checkbox = Checkbox(value=True, label="FP32", interactive=True)
            face_enhance_checkbox = Checkbox(
                value=False, label="Face Enhance", interactive=True
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
