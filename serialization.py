import numpy as np
import OpenEXR
import Imath


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
    fx, fy = 181.45, 181.45
    cx, cy = 96.7, 128.8
    x_coords = np.arange(depth.shape[1])
    y_coords = np.arange(depth.shape[0])
    x_coords, y_coords = np.meshgrid(x_coords, y_coords)
    X = ((x_coords - cx) * depth) / fx
    Y = ((y_coords - cy) * depth) / fy
    depth = np.stack((X, Y, depth), axis=-1)
    return depth
