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
import rembg
import torch
import logging
import argparse
import numpy as np

# import functions
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video


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


# Create a Timer instance
timer = Timer()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Define an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+",
                    help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found,\
        will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model.\
        Could be either a huggingface model id is or a local path. Default:\
        'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering.\
        Smaller chunk size reduces VRAM usage but increases computation time.\
        0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from\
        the input image, and the input image should be an RGB image with gray\
        background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size.\
        Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)

# Parse arguments from the command line
# This will extract the values provided
# for each argument and store them in the args variable
args = parser.parse_args()

# Create the output directory if it doesn't exist
# This is where the results of the script will be saved
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Set the device to use (GPU or CPU)
# If a CUDA-compatible device is available,
# use it; otherwise, use the CPU
device = args.device
if not torch.cuda.is_available():
    device = "cpu"

# Initialize the model
# This will load the pre-trained model and prepare it for use
timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

# Process the input images
# This will load and preprocess each input image
timer.start("Processing images")
images = []

if args.no_remove_bg:

    # If --no-remove-bg is specified, don't remove the background
    rembg_session = None
else:

    # Otherwise, create a new session for removing backgrounds
    rembg_session = rembg.new_session()

for i, image_path in enumerate(args.image):
    if args.no_remove_bg:

        # Load the image as is
        image = np.array(Image.open(image_path).convert("RGB"))
    else:

        # Remove the background and resize the foreground
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + \
            (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

        # Save the preprocessed image
        if not os.path.exists(os.path.join(output_dir, str(i))):
            os.makedirs(os.path.join(output_dir, str(i)))
        image.save(
            os.path.join(
                output_dir,
                str(i),
                f"input.png"
            )
        )
    images.append(image)
timer.end("Processing images")

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    # If --render is specified, render a video
    if args.render:
        timer.start("Rendering")
        render_images = model.render(
            scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(
                output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0],
            os.path.join(output_dir, "render.mp4"), fps=30)
        timer.end("Rendering")

    # Export the mesh
    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes, resolution=args.mc_resolution)
    meshes[0].export(
        os.path.join(
            output_dir, str(i),
            f"mesh.{args.model_save_format}"))
    timer.end("Exporting mesh")
