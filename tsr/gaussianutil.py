"""
This file contains a Gaussian class that uses the GLPN image processor
and depth estimation model to process an image, estimate its depth,
and create a 3D point cloud and mesh from the depth information.

The Gaussian class has the following methods:

set_variables:
sets the input and output paths, foreground ratio,
and padding for the Gaussian object.
gassuin_load:
loads the pre-trained GLPN image processor and depth estimation model.
pre_process:
loads an image from the input path, removes the background,
resizes the foreground, and saves the processed image to the output path.
pre_depth_estimation:
resizes the image to have a height of 512 or less,
and a width proportional to the height.
depth_estimation:
preprocesses the image using the GLPN image processor,
makes predictions using the GLPN depth estimation model,
converts the predicted depth to a numpy array,
and saves the preprocessed image to the output path.
dp_comparison_visual:
crops the original image to match the output size,
and displays the original image and the depth image side by side.
pointcloud:
converts the predicted depth to an 8-bit image,
creates a 3D point cloud from the predicted depth and original image,
and saves the point cloud to the output path.
post_process:
removes outliers from the point cloud,
estimates normals for the point cloud,
visualizes the point cloud with normals,
saves the point cloud to the output path,
creates a mesh from the point cloud using Poisson surface reconstruction,
rotates the mesh by 180 degrees around the x-axis, visualizes the mesh,
and saves the mesh to the output path.

The file also imports the necessary libraries for
image processing and depth estimation,
and sets the matplotlib backend to TkAgg.
"""
# Import necessary libraries
import os
import torch
import rembg
import matplotlib
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from PIL import Image
from tsr.utils import remove_background, resize_foreground

# Import Image processing and depth estimation models
from transformers import GLPNImageProcessor
from transformers import GLPNForDepthEstimation
# Set matplotlib backend to TkAgg
matplotlib.use('TkAgg')


class Gaussian():

    def set_variables(
            self,
            in_path,
            out_path,
            foreground_ratio,
            padding):

        self.in_path = in_path
        self.out_path = out_path
        self.ratio = foreground_ratio
        self.pad = padding

    def gassuin_load(self):
        # Load pre-trained GLPN image processor and depth estimation model
        self.Extractor = GLPNImageProcessor.from_pretrained(
            "vinvino02/glpn-nyu")
        self.Model = GLPNForDepthEstimation.from_pretrained(
            "vinvino02/glpn-nyu")

    def pre_process(self):
        # Load image from path
        image = Image.open(self.in_path)
        rembg_session = rembg.new_session()
        # Remove the background and resize the foreground
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, self.ratio)

        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + \
            (1 - image[:, :, 3:4]) * 0.5
        self.image = Image.fromarray((image * 255.0).astype(np.uint8))
        self.image.save(
            os.path.join(
                f"{self.out_path}images/",
                "Removed_BG.png"
            )
        )
        return image

    def pre_depth_estimation(self):
        image = self.image
        # Resize image to have height of 512 or less,
        # and width proportional to height
        img_H = 512 if image.height > 512 else image.height
        img_H -= (img_H % 32)
        img_W = int(img_H*(image.width / img_H))

        # Adjust width to be a multiple of 32
        diff = img_W % 32
        img_W = img_W - diff if diff < 16 else img_W + 32 - diff
        img_size = (img_W, img_H)
        self.img = image.resize(img_size)

    def depth_estimation(self):
        self.pre_depth_estimation()
        # Preprocess image using GLPN image processor
        input = self.Extractor(images=self.img, return_tensors='pt')

        # Make predictions using GLPN depth estimation model
        with torch.no_grad():
            depth_img = self.Model(**input)
            predicted_depth = depth_img.predicted_depth

        # Convert predicted depth to numpy array and resize to [0, 1000.0]
        depth_img = predicted_depth.squeeze().cpu().numpy() * 1000.0

        # Crop output to remove padding
        pad = self.pad
        self.depth_img = depth_img[pad:-pad, pad:-pad]
        dp_img = Image.fromarray((depth_img).astype(np.uint8))
        # Save the preprocessed image
        if not os.path.exists(os.path.join(self.out_path)):
            os.makedirs(os.path.join(self.out_path))
        dp_img.save(
            os.path.join(
                f"{self.out_path}images/",
                "depth.png"
            )
        )
        return dp_img

    def dp_comparison_visual(self) -> None:
        # Crop original image to match output size
        pad = self.pad
        img = self.img
        depth_img = self.depth_img
        img = img.crop(
            (pad, pad, self.img.width-pad, self.img.height-pad))
        self.img = img

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        ax[0].imshow(img)  # Display image1 on the left
        ax[1].imshow(depth_img, cmap='plasma')  # Display image2 on the right
        plt.show()

    def pointcloud(self):
        # Convert predicted depth to 8-bit image
        img = self.img
        W, H = img.size
        depth_img = self.depth_img
        dp = (depth_img * 255/np.max(depth_img)).astype('uint8')

        # Convert original image to numpy array
        img_arr = np.array(img)

        # Create 3D point cloud from predicted depth and original image
        depth_3d = o3d.geometry.Image(dp)
        img_3d = o3d.geometry.Image(img_arr)
        RGB_dp_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img_3d,
            depth_3d,
            convert_rgb_to_intensity=False
        )

        # Create camera intrinsic parameters
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(W, H, 500, 500, W/2, H/2)

        # Create point cloud from RGBD image
        raw_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            RGB_dp_img,
            camera_intrinsic
        )

        # Visualize point cloud
        o3d.visualization.draw_geometries([raw_pcd])
        o3d.io.write_point_cloud(
            f"{self.out_path}3dfiles/pointcloud_raw.pcd", raw_pcd)
        self.raw_pcd = raw_pcd

    def post_process(self):
        # Remove outliers from point cloud
        cl, ind = self.raw_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=20.0)
        pcd = self.raw_pcd.select_by_index(ind)

        # Estimate normals for point cloud
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()

        # Visualize point cloud with normals
        o3d.visualization.draw_geometries([pcd])

        o3d.io.write_point_cloud(
            f"{self.out_path}3dfiles/pointcloud.pcd", pcd)

        # Create mesh from point cloud using Poisson surface reconstruction
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=10,
            n_threads=1
        )[0]

        # Rotate mesh by 180 degrees around x-axis
        rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        mesh.rotate(rotation, center=(0, 0, 0))

        # Visualize mesh
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        o3d.io.write_point_cloud
        o3d.io.write_triangle_mesh(
            f"{self.out_path}3dfiles/forfront_raw_mesh.obj", mesh)
