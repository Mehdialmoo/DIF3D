
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

    """
    A class to manage the runtime environment and
    perform various tasks related to processing input images and
    generating 3D models using a pre-trained TSR (Triangulated
    Surface Reconstruction) model.

    Attributes:
        timer (Timer): An instance of the Timer class used
        to measure execution time. premodel (Gaussian): An instance
        of the Gaussian class used for pre-processing input images.

        model (TSR):
        An instance of the TSR class used for generating 3D models.
        device (str):
        The device to use for computation (GPU or CPU).
        image_path (str):
        The path to the input images.
        out_path (str):
        The path to save the output files.
        pretrained (str):
        The name of the pre-trained model to use.
        chunk_size (int):
        The chunk size to use for rendering.
        mc_resolution (int):
        The resolution to use for mesh extraction.
        format (str):
        The format to use for saving the 3D mesh.

    Methods:
        set_variables(input_path, output_path, pretrained_model, chunk_size,
            padding, foreground_ratio, mc_resolution, model_save_format):
            Set various variables used by the Runtime class.
        output_address_chk():
            Check and create the necessary output directories.
        processor_check():
            Check and set the device to use for computation.
        initilize():
            Initialize the pre-model and TSR model.
        img_process():
            Process the input images.
        modelRun():
            Run the TSR model on the processed images.
        render():
            Render the 3D model.
        export_mesh():
            Export the 3D mesh in the specified format.
    """


class Runtime():

    def __init__(self):
        # Initialize the timer
        self.timer = Timer()
        # Initialize the pre-model
        self.premodel = Gaussian()
        # Initialize the TSR model (None for now)
        # based on pretrainde or to be trained
        self.model = None
        # Set up logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    def set_variables(
            self,
            input_path="input/",
            output_path="output/",
            pretrained_model="stabilityai/TripoSR",
            chunk_size=8192,
            padding=16,
            foreground_ratio=0.85,
            mc_resolution=256,
            model_save_format="obj"
    ):
        # Set the input image path
        self.image_path = input_path
        # Set the output path
        self.out_path = output_path
        # Set the pre-trained model name
        self.pretrained = pretrained_model
        # Set the chunk size for rendering
        self.chunk_size = chunk_size
        # Set the resolution for mesh extraction
        self.mc_resolution = mc_resolution
        # Set the format for saving the 3D mesh
        self.format = model_save_format

        # Set variables for the pre-model
        self.premodel.set_variables(
            in_path=input_path,
            out_path=output_path,
            foreground_ratio=foreground_ratio,
            padding=padding)
        # Check and create the necessary output directories
        self.output_address_chk()
        # Check and set the processor either GPU (CUDA) or CPU
        self.processor_check()

    def output_address_chk(self):
        # creates directory for output results
        os.makedirs(self.out_path, exist_ok=True)
        os.makedirs(f"{self.out_path}images/", exist_ok=True)
        os.makedirs(f"{self.out_path}renderfiles/", exist_ok=True)
        os.makedirs(f"{self.out_path}3dfiles/", exist_ok=True)

    def processor_check(self):
        # Set the device to use (GPU or CPU)
        # If a CUDA-compatible device is available,
        # use it; otherwise, use the CPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def initilize(self):
        # Initialize the model
        # This will load the pre-trained model and prepare it for use
        self.timer.start("Initializing pre-model(Gaussian model)")
        self.premodel.gassuin_load()
        self.timer.end("Initializing pre-model(Gaussian model)")
        self.timer.start("Initializing TSR (diffsusion 3D) model")
        self.model = TSR.from_pretrained(
            self.pretrained,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self.timer.end("Initializing TSR (diffsusion 3D) model")

    def img_process(self):
        # Start a timer for the image processing step
        self.timer.start("Processing images")
        # Load and preprocess the input image using
        # the pre_process method of the premodel object
        self.image = self.premodel.pre_process()
        # Estimate the depth of the input image using the
        # depth_estimation method of the premodel object
        self.depth_image = self.premodel.depth_estimation()
        # Perform a depth prediction comparison visualization
        # using the dp_comparison_visual method of the premodel object
        self.premodel.dp_comparison_visual()
        # End the timer for the image processing step
        self.timer.end("Processing images")

    def modelRun(self):
        # Log a message to inform the user that
        # the process might take a few minutes
        logging.info("please wait this process might take a few minutes...")
        # Start a timer for the model running step
        self.timer.start("Running model")
        # Generate a point cloud
        self.premodel.pointcloud()
        # clean the pointcloud and create a forfront mesh
        # for next model hologan and zero1-to-3 to create mesh
        self.premodel.post_process()
        # Run the model on the input image with no gradient computation
        with torch.no_grad():
            self.scene_codes = self.model([self.image], device=self.device)
        # End the timer for the model running step
        self.timer.end("Running model")

    def render(self):
        # Start a timer to track the rendering time
        self.timer.start("Rendering")
        # Render images using the model, with 30 views, and return PIL images
        render_images = self.model.render(
            self.scene_codes, n_views=30, return_type="pil")
        # Iterate over the rendered images
        for ri, render_image in enumerate(render_images[0]):
            # Save each image to a file
            # with a numbered filename (e.g. render_001.png)
            render_image.save(os.path.join(
                f"{self.out_path}renderfiles/", f"render_{ri:03d}.png"))
            # Save a video using all the rendered images
            save_video(render_images[0], os.path.join(
                f"{self.out_path}renderfiles/", "render.mp4"), fps=30)
            # End the timer (this should be done after the loop, not inside it)
            self.timer.end("Rendering")

    def export_mesh(self):
        # Start a timer to track the mesh export time
        self.timer.start("Exporting mesh")
        # Extract the mesh using the model, with the specified resolution
        meshes = self.model.extract_mesh(
            self.scene_codes, resolution=self.mc_resolution)
        # Export the first mesh to a file with the specified format
        meshes[0].export(
            os.path.join(
                f"{self.out_path}3dfiles/",
                f"mesh.{self.format}"))
        # End the timer
        self.timer.end("Exporting mesh")
