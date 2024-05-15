"""
This file defines a Text-Driven 3D Shape Generation and
Rendering model (difussion 3D) using PyTorch.

The difussion 3D model consists of several components:
1. Image Tokenizer:
    Tokenizes the input image into a sequence of tokens.
2. Tokenizer:
    Encodes the input sequence of tokens into a fixed-length vector.
3. Backbone:
    Processes the input sequence of tokens and generates a scene code.
4. Post Processor:
    Post-processes the scene code to generate a set of scene parameters.
5. Decoder:
    Decodes the scene parameters and generates a 3D shape.
6. Renderer:
    Renders the 3D shape into a 2D image.

The file also includes several utility functions and classes for:
1. Preprocessing the input image.
2. Generating spherical cameras for rendering.
3. Extracting meshes using marching cubes.
4. Loading pre-trained models from a given path or name.

The file defines a `difussion 3D` class that implements
the stable diffusion 3d model. The `diff3rd` class has the following methods:

1. `from_pretrained`: Loads a pre-trained model from a given path or name.
2. `configure`: Configures the model by initializing the image tokenizer,
tokenizer, backbone, post processor, decoder, and renderer.
3. `forward`: Performs a forward pass through the model.
4. `render`: Renders the scene codes into 2D images.
5. `set_marching_cubes_resolution`: Sets the marching cubes resolution.
6. `extract_mesh`: Extracts meshes from the scene codes using marching cubes.

The file also includes a `BaseModule` class that provides
a common interface for all the components of the difussion 3D model.
"""

# Import necessary libraries
import os
import numpy as np
import PIL.Image
import torch
import trimesh

from dataclasses import dataclass
from typing import List, Union
from einops import rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from PIL import Image

# Import custom modules
from tsr.models.isosurface import MarchingCubeHelper
from tsr.utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)

# Define the diff3rd: diffusion 3d class


class TSR(BaseModule):
    # Define the Config class
    @dataclass
    class Config(BaseModule.Config):
        # Configuration parameters
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    # Initialize the TSR class with a configuration
    cfg: Config

    @classmethod
    # Load a pre-trained model from a given path or name
    def from_pretrained(
        cls, pretrained_model_name_or_path: str,
        config_name: str, weight_name: str
    ):
        # Load the configuration and weights
        # Check if the path is a directory or a model name
        if os.path.isdir(pretrained_model_name_or_path):
            # Load the configuration and weights from the directory
            config_path = os.path.join(
                pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(
                pretrained_model_name_or_path, weight_name)
        else:
            # Load the configuration and weights from the Hugging Face Hub
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=config_name
            )
            weight_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename=weight_name
            )
        # Load the configuration and weights
        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        ckpt = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(ckpt)
        return model

    # Configure the model
    def configure(self):
        # Initialize the image tokenizer, tokenizer,
        # backbone, post processor, decoder, and renderer
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    # Forward pass through the model
    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        # Preprocess the input image
        rgb_cond = self.image_processor(
            image,
            self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        # Tokenize the input image
        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        # Generate tokens
        tokens: torch.Tensor = self.tokenizer(batch_size)

        # Pass tokens through the backbone
        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        # Post-process the tokens
        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    # Render the scene codes
    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        # Get the origin and direction of the rays for each camera
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        # Move the rays to the same device as the scene codes
        rays_o, rays_d = rays_o.to(
            scene_codes.device), rays_d.to(scene_codes.device)

        # Define a function to process output image based on the return type
        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        # Initialize an empty list to store the rendered images
        images = []
        # Iterate over each scene code
        for scene_code in scene_codes:
            # Initialize an empty list to store the images for each view
            images_ = []
            # Iterate over each view
            for i in range(n_views):
                # Render images for the current view with no gradient
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                # Process the output image based on
                # the return type and append it to the list
                images_.append(process_output(image))
            # Append a list of images for each view to a list of all images
            images.append(images_)

        # Return the list of all images
        return images

    # Set the marching cubes resolution
    def set_marching_cubes_resolution(self, resolution: int):
        # Check if the marching cubes helper is already initialized
        # and has the same resolution
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        # Initialize the marching cubes helper with the given resolution
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(
            self,
            scene_codes,
            resolution: int = 256,
            threshold: float = 25.0):
        # Set the marching cubes resolution
        self.set_marching_cubes_resolution(resolution)
        # Initialize an empty list to store the meshes
        meshes = []
        # Iterate over the scene codes
        for scene_code in scene_codes:
            # Query the density of the scene code
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(
                            scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]
            # Extract the mesh using marching cubes
            v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
            v_pos = scale_tensor(
                v_pos,
                self.isosurface_helper.points_range,
                (-self.renderer.cfg.radius, self.renderer.cfg.radius),
            )
            # Query the color of the mesh
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]

            # Create a Trimesh object from the vertices, faces, and colors
            mesh = trimesh.Trimesh(
                vertices=v_pos.cpu().numpy(),
                faces=t_pos_idx.cpu().numpy(),
                vertex_colors=color.cpu().numpy(),
            )
            # Add the mesh to the list
            meshes.append(mesh)
        # Return the list of meshes
        return meshes
