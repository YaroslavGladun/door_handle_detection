import numpy as np
import OpenEXR
import Imath

from common import CameraIntrinsics


def read_exr(file_path: str) -> np.ndarray:
    exr_file = OpenEXR.InputFile(file_path)

    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])

    return depth


def read_exr_points(file_path: str) -> np.ndarray:
    depth = read_exr(file_path)
    intrinsics = CameraIntrinsics.source_depth_image_intrinsics()
    x_coords = np.arange(depth.shape[1])
    y_coords = np.arange(depth.shape[0])
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    depth = np.stack((x_coords, y_coords, depth), axis=-1)
    return intrinsics.uvd_to_xyz(depth)
