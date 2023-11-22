BASE_PATH = "EXR_RGBD"

import json
import cv2
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import math
from typing import Tuple


def read_exr(file_path):
    exr_file = OpenEXR.InputFile(file_path)

    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the depth channel (assuming it's named 'Z')
    depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])  # Reshape to the image size

    return depth


def draw_origin_on_image(image, cam_info, t, r, transparency):
    fx, fy = cam_info[0, 0], cam_info[1, 1]
    px, py = cam_info[0, 2], cam_info[1, 2]
    # a0 = np.array([0, 0, 0.3])
    ax = np.array([0.05, 0, 0]) + t
    ay = np.array([0, 0.05, 0]) + t
    az = np.array([0, 0, 0.05]) + t

    a0_cv_pt = (int(fx * t[0] / t[2] + px), int(fy * t[1] / t[2] + py))
    ax_cv_pt = (int(fx * ax[0] / ax[2] + px), int(fy * ax[1] / ax[2] + py))
    ay_cv_pt = (int(fx * ay[0] / ay[2] + px), int(fy * ay[1] / ay[2] + py))
    az_cv_pt = (int(fx * az[0] / az[2] + px), int(fy * az[1] / az[2] + py))

    new_image = image.copy()
    cv2.line(new_image, a0_cv_pt, ax_cv_pt, (0, 0, 255), 15)
    cv2.line(new_image, a0_cv_pt, ay_cv_pt, (0, 255, 0), 15)
    cv2.line(new_image, a0_cv_pt, az_cv_pt, (255, 0, 0), 15)
    cv2.addWeighted(image, transparency, new_image, 1 - transparency, 0, new_image)

    return new_image


def ijd_to_xyz(i, j, d, cam_info):
    # Extract the focal lengths and principal point coordinates from cam_info
    fx, fy = cam_info[0, 0], cam_info[1, 1]
    px, py = cam_info[0, 2], cam_info[1, 2]

    # Convert from image coordinates (i, j, d) to 3D world coordinates (X, Y, Z)
    X = (i - px) * d / fx
    Y = (j - py) * d / fy
    Z = d

    return np.array([X, Y, Z])


# metadata_path = f"{BASE_PATH}/metadata_x.json"
metadata_path = f"metadata.json"

with open(metadata_path, 'r') as file:
    metadata = json.load(file)

print(metadata.keys())
print(metadata['initPose'])
print(metadata['K'])
camera_matrix = np.array(metadata['K']).reshape(3, 3).T
dist_coeffs = np.array([2.03226134e-01, -7.04873546e-01, -3.51671357e-05, -8.49750941e-04, 6.28718638e-01])
handle_pose = ijd_to_xyz(595, 718, 0.37, camera_matrix)

images = []
# for i in range(len(metadata["frameTimestamps"])):
# print(d[128, 96])
# exit(0)

# Last three values.
# 5 of camera look top
# 4 look right
poses = np.array(metadata['poses'])
for i in range(7):
    plt.plot(poses[:, i], label=f"{i}")
plt.legend()
plt.show()
# poses[:, 4] = -poses[:, 4]
# poses[:, 5] = -poses[:, 5]
for i in range(1000):
    rgb_image_path = f"{BASE_PATH}/rgb/{i}.jpg"
    image = cv2.imread(rgb_image_path)
    depth_image = read_exr(f"{BASE_PATH}/depth/{i}.exr")
    # plt.imshow(cv2.resize(depth_image, (images[-1].shape[1], images[-1].shape[0])))
    # plt.show()

    pose = handle_pose + poses[0][-3:] - poses[i][-3:]
    # pose = np.array([0, 0, 0]) - metadata['poses'][i][:3]
    image_with_axes = draw_origin_on_image(
        image,
        camera_matrix,
        pose,
        None,
        0.5
    )
    #
    # # Display the image
    cv2.imshow('Image with 3D Axes', image_with_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
