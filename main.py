from gradio import Blocks, Button, Code, Dropdown, File, Gallery, Image, Label, Plot, Row, Tab, Slider
from gradio.themes import Monochrome

with Blocks(title="AAML WebUI", theme=Monochrome()) as application:
    with Tab("Train"):
        dataset_field = File(
            label="Choose Dataset",
            type="file",
            file_types=[".pickle"],
            interactive=True
        )
        model_dropdown = Dropdown(
            choices=["Test_1", "Test_2"],
            value="Test_1",
            label="Models",
            interactive=True
        )
        with Row():
            epoch_slider = Slider(1, 10000, value=20, step=1, label="Epochs", interactive=True)
            batch_slider = Slider(0, 4096, step=1, label="Batch Size", interactive=True)
        with Row():
            train_plot = Plot(label="Train Plot")
            test_plot = Plot(label="Test Plot")
        train_button = Button(value="Train Button")
    with Tab("Test"):
        with Row():
            test_image = Image(
                image_mode="RGBA",
                source="upload",
                type="pil"
            )
            output_image = Gallery(label="Just DL Model Output Image")
            upscale_output_image = Gallery(label="With Upscale Method Output Image")
        with Row():
            tile_size_slider = Slider(
                0, 1024, value=192, step=1, label="Tile Size", interactive=True 
            )
            upscale_model_dropdown = Dropdown(
                choices=['RealESRGAN_x4plus',
                        'RealESRNet_x4plus',
                        'RealESRGAN_x4plus_anime_6B',
                        'RealESRGAN_x2plus',
                        'realesr-general-x4v3'],
                value='RealESRGAN_x4plus',
                label="Models",
                interactive=True
            )
        test_button = Button(value="Run")
    with Tab("Create"):
        custom_model_code = Code(
            value="Hello World",
            language="python",
            lines=25,
            interactive=True
        )
        save_custom_model = Button("Save Model")

if __name__ == "__main__":
    application.queue(concurrency_count=4).launch()