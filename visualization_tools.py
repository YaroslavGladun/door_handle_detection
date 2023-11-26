import numpy as np
import cv2
from common import CameraIntrinsics, get_box_vertices


def draw_origin_on_image(image: np.ndarray, pose: np.ndarray,
                         intrinsics=CameraIntrinsics.source_rgb_image_intrinsics()):
    fx, fy = intrinsics.fx, intrinsics.fy
    px, py = intrinsics.cx, intrinsics.cy
    a0 = pose[:-1, 3]
    ax = np.dot(pose, np.array([0.02, 0, 0, 1]).reshape(-1, 1))[:-1]
    ay = np.dot(pose, np.array([0, 0.02, 0, 1]).reshape(-1, 1))[:-1]
    az = np.dot(pose, np.array([0, 0, 0.02, 1]).reshape(-1, 1))[:-1]

    a0_cv_pt = (int(fx * a0[0] / a0[2] + px), int(fy * a0[1] / a0[2] + py))
    ax_cv_pt = (int(fx * ax[0] / ax[2] + px), int(fy * ax[1] / ax[2] + py))
    ay_cv_pt = (int(fx * ay[0] / ay[2] + px), int(fy * ay[1] / ay[2] + py))
    az_cv_pt = (int(fx * az[0] / az[2] + px), int(fy * az[1] / az[2] + py))

    cv2.line(image, a0_cv_pt, ax_cv_pt, (0, 0, 255), 3)  # Red for X-axis
    cv2.line(image, a0_cv_pt, ay_cv_pt, (0, 255, 0), 3)  # Green for Y-axis
    cv2.line(image, a0_cv_pt, az_cv_pt, (255, 0, 0), 3)  # Blue for Z-axis

    return image


def draw_box_on_image(image: np.ndarray, pose: np.ndarray, box_size: list,
                      intrinsics=CameraIntrinsics.source_rgb_image_intrinsics()):
    box_vertices = get_box_vertices([0, 0, 0] + box_size)
    box_vertices = np.dot(pose, np.concatenate((box_vertices.T, np.ones((1, 8))), axis=0))[:-1].T
    box_vertices = intrinsics.xyz_to_uvd(box_vertices)
    lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    for v1, v2 in lines:
        cv2.line(image, (int(box_vertices[v1, 0]), int(box_vertices[v1, 1])),
                 (int(box_vertices[v2, 0]), int(box_vertices[v2, 1])), (0, 255, 0), 3)
    return image


def draw_box_on_image_by_vertices(image: np.ndarray, box_vertices: np.ndarray,
                      intrinsics=CameraIntrinsics.source_rgb_image_intrinsics()):
    box_vertices = intrinsics.xyz_to_uvd(box_vertices)
    lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    for v1, v2 in lines:
        cv2.line(image, (int(box_vertices[v1, 0]), int(box_vertices[v1, 1])),
                 (int(box_vertices[v2, 0]), int(box_vertices[v2, 1])), (0, 255, 0), 3)
    return image

# im = cv2.imread('/home/yaroslav/Desktop/door_handle_detection/dataset/rgb_30c71e67e1d2e46c268fba41687e1d3b.jpg')
# pose = np.load('/home/yaroslav/Desktop/door_handle_detection/dataset/pose_30c71e67e1d2e46c268fba41687e1d3b.npy')
# im = draw_box_on_image(im, pose, [0.15, 0.02, 0.05])
# im = draw_origin_on_image(im, pose)
# im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
# cv2.imshow('img', im)
# cv2.waitKey(0)
