import hashlib
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from random import shuffle
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from common import Plane, RotationMatrix, find_bounding_box_with_max_points_inside, get_box_vertices
from serialization import read_exr_points
from visualization_tools import draw_box_on_image_by_vertices


def create_plane_from_equation(plane: Plane, size=10, segments=10):
    a, b, c, d = plane
    if c == 0:
        raise ValueError("c cannot be zero for this implementation.")

    # Generate grid of points
    x = np.linspace(-size / 2, size / 2, segments)
    y = np.linspace(-size / 2, size / 2, segments)
    x, y = np.meshgrid(x, y)
    z = -(a * x + b * y + d) / c

    # Create vertices
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Create faces
    faces = []
    for i in range(segments - 1):
        for j in range(segments - 1):
            idx = i * segments + j
            faces.append([idx, idx + segments, idx + 1])
            faces.append([idx + 1, idx + segments, idx + segments + 1])

    # Create plane mesh
    plane_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                           triangles=o3d.utility.Vector3iVector(faces))
    plane_mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Set color

    return plane_mesh


def draw_points_o3d(points: np.ndarray, verc: np.ndarray, plane: Plane):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # box = o3d.geometry.TriangleMesh.create_box(width=box_size[0], height=box_size[1], depth=box_size[2])
    # box.translate(bb[:3])
    lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    lines_visuals = []
    for v1, v2 in lines:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([verc[v1], verc[v2]])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))
        lines_visuals.append(line)
    plane = create_plane_from_equation(plane, size=10, segments=10)
    o3d.visualization.draw_geometries([pcd, coordinate_frame, plane] + lines_visuals)


depth_images_path = "EXR_RGBD/depth"
rgb_images_path = "EXR_RGBD/rgb"
dataset_directory = "dataset"
generated_dataset_directory = "generated_dataset"
dbscan = DBSCAN(eps=0.02, min_samples=3)
ransac = RANSACRegressor()

files = os.listdir(depth_images_path)
shuffle(files)
files = files[:300]
for entry in tqdm(files):
    entry_name, _ = os.path.splitext(entry)
    depth_image_path = f"{depth_images_path}/{entry_name}.exr"
    rgb_image_path = f"{rgb_images_path}/{entry_name}.jpg"
    depth = read_exr_points(depth_image_path)
    points = depth.reshape(-1, 3)
    points[np.isnan(points).any(axis=1)] = 0
    rgb = cv2.imread(rgb_image_path)
    new_name = str(hashlib.md5(rgb.tobytes()).hexdigest())

    ransac.fit(points[:, :2], points[:, 2])
    (a, b), c = ransac.estimator_.coef_, -1
    d = ransac.estimator_.intercept_
    plane = Plane.normalized(a, b, c, d)
    distances = plane.distance_to_points(depth)
    mask = (distances > 0.02) & (np.linalg.norm(depth, axis=-1) < 1)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    filtered_points = points[mask.reshape(-1)]
    if filtered_points.shape[0] < 100:
        continue

    plane_rotation = RotationMatrix.from_vector(plane.norm())
    rotated_to_plane_points = (plane_rotation.T @ filtered_points.T).T

    dbscan.fit(rotated_to_plane_points)
    labels = dbscan.labels_
    max_label = max(labels)
    box_angle = []
    for z_angle in np.linspace(0, 2 * np.pi, 100):
        rotation = RotationMatrix.from_angle_axis(z_angle, [0, 0, 1])
        rotated_filtered_points = (rotation.T @ rotated_to_plane_points.T).T
        for label in range(max_label + 1):
            cluster_points = rotated_filtered_points[labels == label]
            if len(cluster_points) < 100:
                continue
            box_min = np.min(cluster_points, axis=0)
            box_max = np.max(cluster_points, axis=0)
            box_angle.append((np.concatenate([box_min, box_max], axis=-1), z_angle))
    target_box_size = np.array([0.15, 0.02, 0.05])
    scores = []
    if len(box_angle) == 0:
        continue
    for box, z_angle in box_angle:
        box_size = box[3:] - box[:3]
        score = np.sum(np.abs(box_size - target_box_size))
        scores.append(score)
    best_box, best_z_angle = box_angle[np.argmin(scores)]
    rotation = RotationMatrix.from_angle_axis(-best_z_angle, [0, 0, 1])
    box_vertices = get_box_vertices(best_box)
    center_point = np.mean(box_vertices, axis=0)
    center_point[-1] = center_point[-1] - 0.02
    best_box = np.concatenate([center_point - target_box_size / 2, center_point + target_box_size / 2], axis=-1)
    box_vertices = get_box_vertices(best_box)
    box_vertices = (rotation.T @ box_vertices.T).T
    box_vertices_in_camera = (plane_rotation @ box_vertices.T).T

    np.save(f"{generated_dataset_directory}/depth_{new_name}.npy", depth)
    np.save(f"{generated_dataset_directory}/box_{new_name}.npy", box_vertices_in_camera)
    np.save(f"{generated_dataset_directory}/plane_{new_name}.npy", plane)
    cv2.imwrite(f"{generated_dataset_directory}/rgb_{new_name}.jpg", rgb)
    rgb = draw_box_on_image_by_vertices(rgb, box_vertices_in_camera)
    rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))
    cv2.imwrite(f"{generated_dataset_directory}/visualization_{new_name}.jpg", rgb)

    # cv2.imshow('img', rgb)
    # cv2.waitKey(0)
    # draw_points_o3d(points, box_vertices_in_camera, plane)
    # angle_score_bb.sort(key=lambda x: x[1], reverse=True)
    # z_angle, score, bb = angle_score_bb[0]
    # rotation = RotationMatrix.from_angle_axis(z_angle, [0, 0, 1])
    # rotated_filtered_points = (rotation.T @ filtered_points.T).T
    # draw_points_o3d(rotated_filtered_points, bb)
    # print(bb)

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
