"""
Depth Estimation and 3D Reconstruction using GLPN and Open3D.

The following script demonstrates depth estimation,
from a single image using the GLPN depth estimation model
and subsequent 3D reconstruction using Open3D.

firstly load pre-trained GLPN image processor and depth estimation
model.

Preprocess the image and make predictions using
the GLPN depth estimation model. Convert the predicted depth to
a numpy array and resize it to [0, 1000.0]. Crop the output to
remove padding and crop the original image to match the output size.

Visualize the original image and predicted depth using matplotlib.

Convert the predicted depth to an 8-bit image and
the original image to a numpy array. Create a 3D point cloud
from the predicted depth and original image using Open3D.

Create camera intrinsic parameters and create
a point cloud from the RGBD image using Open3D. Visualize the point cloud.

Remove outliers from the point cloud and estimate normals for the point cloud.
Visualize the point cloud with normals.

Create a mesh from the point cloud using Poisson surface reconstruction with
Open3D. Rotate the mesh by 180 degrees around the x-axis and
visualize the resulting mesh.

"""
# Import necessary libraries
import torch
import matplotlib
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from PIL import Image

# Import Image processing and depth estimation models
from transformers import GLPNImageProcessor
from transformers import GLPNForDepthEstimation


class gaussian():

    def __init__(self) -> None:

        # Set matplotlib backend to TkAgg
        matplotlib.use('TkAgg')

    def set_gaussian_var(self, path, pad_size):
        # Set path to image ()
        self.path = path  # "D:\git ex\MC\pic.jpg"
        self.pad = pad_size  # 16

    def gassuin_load(self):
        # Load pre-trained GLPN image processor and depth estimation model
        self.Extractor = GLPNImageProcessor.from_pretrained(
            "vinvino02/glpn-nyu")
        self.Model = GLPNForDepthEstimation.from_pretrained(
            "vinvino02/glpn-nyu")

    def Pre_process(self):
        # Load image from path
        image = Image.open(self.path)

        # Resize image to have height of 512 or less,
        # and width proportional to height
        img_H = 512 if image.height > 512 else image.height
        img_H -= (img_H % 32)
        img_W = int(img_H*(image.width / img_H))

        # Adjust width to be a multiple of 32
        diff = img_W % 32
        img_W = img_W - diff if diff < 16 else img_W + 32 - diff
        img_size = (img_W, img_H)
        self.img = Image.resize(img_size)

    def depth_estimation(self) -> None:

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

    def dp_comparison_visual(self) -> None:

        # Crop original image to match output size
        pad = self.pad
        img = self.img
        depth_img = self.depth_img
        img = img.crop(
            (pad, pad, self.img.width-pad, self.img.height-pad))
        self.img = img

        # Visualize original image and predicted depth
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)

        plt.subplot(1, 2, 2)
        plt.imshow(depth_img, cmap='plasma')
        plt.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)

        plt.tight_layout()

    def o3rd_estimator(self):
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

        # Remove outliers from point cloud
        cl, ind = raw_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=20.0)
        pcd = raw_pcd.select_by_index(ind)

        # Estimate normals for point cloud
        pcd.estimate_normals()
        pcd.orient_normals_to_align_with_direction()

        # Visualize point cloud with normals
        o3d.visualization.draw_geometries([pcd])

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
