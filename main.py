import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from random import shuffle
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from common import Plane, RotationMatrix, find_bounding_box_with_max_points_inside
from serialization import read_exr_points


def draw_points_o3d(points: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])


depth_images_path = "EXR_RGBD/depth"
rgb_images_path = "EXR_RGBD/rgb"
dataset_directory = "dataset"
ransac = RANSACRegressor(LinearRegression(), min_samples=3, residual_threshold=0.02, max_trials=1000)

files = os.listdir(depth_images_path)
shuffle(files)
for entry in files:
    entry_name, _ = os.path.splitext(entry)
    depth_image_path = f"{depth_images_path}/{entry_name}.exr"
    rgb_image_path = f"{rgb_images_path}/{entry_name}.jpg"
    depth = read_exr_points(depth_image_path)
    points = depth.reshape(-1, 3)
    rgb = cv2.imread(rgb_image_path)

    ransac.fit(points[:, :2], points[:, 2])
    (a, b), c = ransac.estimator_.coef_, -1
    d = ransac.estimator_.intercept_
    plane = Plane.normalized(a, b, c, d)
    distances = plane.distance_to_points(depth)
    mask = distances > 0.02
    plt.imshow(mask, cmap='gray')
    plt.show()

    filtered_points = points[mask.reshape(-1)]

    rotation = RotationMatrix.from_vector(plane.norm())
    draw_points_o3d((rotation.T @ filtered_points.T).T)
    find_bounding_box_with_max_points_inside(filtered_points, rotation)

    # new_name = str(hashlib.md5(depth.tobytes()).hexdigest())
    #
    # labler = LabelTestCase(rgb, depth)
    # affine = labler.start()
    # plane = labler.plane()
    # distances = plane.distance_to_points(depth)
    # depth_copy = depth.copy()
    # points = depth_copy[distances < 0.01]
    # plt.imshow(depth_copy[..., 2])
    # plt.show()
    # corrector = AffineCorrector(rgb, depth, affine)
    # affine = corrector.start()
    # print(affine)
    #
    # np.save(f"{dataset_directory}/depth_{new_name}.npy", depth)
    # np.save(f"{dataset_directory}/pose_{new_name}.npy", affine)
    # cv2.imwrite(f"{dataset_directory}/rgb_{new_name}.jpg", rgb)
    # os.remove(rgb_image_path)
    # os.remove(depth_image_path)
