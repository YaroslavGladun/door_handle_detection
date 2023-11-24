import numpy as np
import cv2
from common import CameraIntrinsics


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
