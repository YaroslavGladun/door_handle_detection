import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from random import shuffle
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from common import Plane, RotationMatrix, find_bounding_box_with_max_points_inside
from serialization import read_exr_points


def draw_points_o3d(points: np.ndarray, bb: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    box_size = bb[3:] - bb[:3]
    box = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])
    box.translate(bb[:3])
    o3d.visualization.draw_geometries([pcd, coordinate_frame, box])


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
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    filtered_points = points[mask.reshape(-1)]

    rotation = RotationMatrix.from_vector(plane.norm())
    rotated_filtered_points = (rotation.T @ filtered_points.T).T

    angle_score_bb = []
    for z_angle in tqdm(np.linspace(0, 2 * np.pi, 100)):
        rotation = RotationMatrix.from_angle_axis(z_angle, [0, 0, 1])
        rotated_filtered_points = (rotation.T @ filtered_points.T).T
        bb, score = find_bounding_box_with_max_points_inside(
            rotated_filtered_points, [0.15, 0.03, 0.05])
        angle_score_bb.append((z_angle, score, bb))
    angle_score_bb.sort(key=lambda x: x[1], reverse=True)
    z_angle, score, bb = angle_score_bb[0]
    rotation = RotationMatrix.from_angle_axis(z_angle, [0, 0, 1])
    rotated_filtered_points = (rotation.T @ filtered_points.T).T
    draw_points_o3d(rotated_filtered_points, bb)
    print(bb)

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
