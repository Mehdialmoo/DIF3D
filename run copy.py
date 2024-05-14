"""
This script uses a pre-trained model to extract 3D meshes
from input images and optionally render a video of
the resulting mesh. The script can process multiple images in
a single run and saves the results in a specified output directory.

The script defines a Timer class to measure execution time and
creates a Timer instance to time various parts of the script.

The script parses command line arguments to configure various options, such as
the input images, device to use, pretrained model path, chunk size, marching
cubes resolution, whether to remove the background, foreground ratio, output
directory, mesh save format, and whether to render a video.

The script then initializes the model, processes the input images,
and runs the model on each image to extract the 3D mesh.
If the --render option is specified, the script renders a video of the mesh.
Finally, the script exports the mesh in the specified format and saves
it to the output directory.
"""

# Import necessary libraries
import os
import time
import torch
import logging

# import functions
from tsr.system import TSR
from tsr.gaussianutil import Gaussian
from tsr.utils import save_video


# Define a Timer class to measure execution time
class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        # Start timing a task
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        # End timing a task and log the execution time
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


class Runtime():
    def __init__(self):
        # Create a Timer instance
        self.timer = Timer()
        self.model = TSR()
        self.premodel = Gaussian()
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    def set_output_address(self, path):
        os.makedirs(path, exist_ok=True)

    def processor_check(self):
        # Set the device to use (GPU or CPU)
        # If a CUDA-compatible device is available,
        # use it; otherwise, use the CPU
        self.device = "gpu" if torch.cuda.is_available() else "cpu"

    def set_variables(
            self,
            input_path,
            output_path,
            pretrained_model,
            chunk_size,
            padding,
            foreground_ratio,
            mc_resolution,
            model_save_format
    ):
        self.image_path = input_path
        self.out_path = output_path
        self.pretrained = pretrained_model,
        self.chunk_size = chunk_size
        self.mc_resolution = mc_resolution
        self.format = model_save_format
        self.premodel.set_variables(
            path=input_path,
            foreground_ratio=foreground_ratio,
            padding=padding)

    def initilize(self):
        # Initialize the model
        # This will load the pre-trained model and prepare it for use
        self.timer.start("Initializing pre-model(Gaussian model)")
        self.premodel.gassuin_load()
        self.timer.end("Initializing pre-model(Gaussian model)")
        self.timer.start("Initializing TSR (diffsusion 3D) model")
        self.model.from_pretrained(
            self.pretrained,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self.timer.end("Initializing TSR (diffsusion 3D) model")

    def img_process(self):
        # Process the input images
        # This will load and preprocess each input image
        self.timer.start("Processing images")
        self.images_lst = []
        for i, image_path in enumerate(self.image_path):
            image = self.premodel.pre_process(image_path)
            depth_image = self.premodel.depth_estimation()

            # Save the preprocessed image
            if not os.path.exists(os.path.join(self.out_path)):
                os.makedirs(os.path.join(self.out_path))
            image.save(
                os.path.join(
                    self.out_path,
                    str(i),
                    "Removed_BG.png"
                )
            )
            self.images_lst.append(image)
        self.timer.end("Processing images")

    def modelRun(self):
        logging.info("Running image ...")
        self.timer.start("Running model")
        with torch.no_grad():
            self.scene_codes = self.model([self.image], device=self.device)
        self.timer.end("Running model")

    def render(self):
        self.timer.start("Rendering")
        render_images = self.model.render(
            self.scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(
                self.out, f"render_{ri:03d}.png"))
            save_video(render_images[0], os.path.join(
                self.out_path, "render.mp4"), fps=30)
            self.timer.end("Rendering")

    def Export_mesh(self):
        # Export the mesh
        self.timer.start("Exporting mesh")
        meshes = self.model.extract_mesh(
            self.scene_codes, resolution=self.mc_resolution)
        meshes[0].export(
            os.path.join(
                self.out_path,
                f"mesh.{self.format}"))
        self.timer.end("Exporting mesh")
