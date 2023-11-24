import numpy as np
from typing import Tuple
from tqdm import tqdm


class Vector3(np.ndarray):

    @staticmethod
    def unit_x():
        return Vector3(np.array([1, 0, 0]))

    @staticmethod
    def unit_y():
        return Vector3(np.array([0, 1, 0]))

    @staticmethod
    def unit_z():
        return Vector3(np.array([0, 0, 1]))

    @staticmethod
    def normalized(a, b, c):
        p = Vector3(np.array([a, b, c]))
        return p / np.linalg.norm(p)

    def __new__(cls, input_array: np.ndarray):
        if len(input_array) != 3:
            raise ValueError("Point must have a size of 3")

        obj = np.asarray(input_array).view(cls)
        return obj


class RotationMatrix(np.ndarray):

    def __new__(cls, input_array: np.ndarray):
        if input_array.shape != (3, 3):
            raise ValueError("Rotation matrix must have a size of 3x3")

        obj = np.asarray(input_array).view(cls)
        return obj

    @staticmethod
    def from_vector(vector: Vector3):
        x, y, z = vector
        c = np.cos
        s = np.sin
        matrix = np.array([
            [c(y) * c(z), c(z) * s(x) * s(y) - c(x) * s(z), c(x) * c(z) * s(y) + s(x) * s(z)],
            [c(y) * s(z), c(x) * c(z) + s(x) * s(y) * s(z), c(x) * s(y) * s(z) - c(z) * s(x)],
            [-s(y), c(y) * s(x), c(x) * c(y)]
        ])
        return RotationMatrix(matrix)

    @staticmethod
    def from_angle_axis(angle: float, axis: Vector3):
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        matrix = np.array([
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]
        ])
        return RotationMatrix(matrix)


class Plane(np.ndarray):

    @staticmethod
    def normalized(a, b, c, d):
        abc_norm = np.linalg.norm([a, b, c])
        a, b, c, d = a / abc_norm, b / abc_norm, c / abc_norm, d / abc_norm
        return Plane(np.array([a, b, c, d]))

    def __new__(cls, input_array: np.ndarray):
        if len(input_array) != 4:
            raise ValueError("Plane must have a size of 4")

        obj = np.asarray(input_array).view(cls)
        return obj

    def norm(self) -> Vector3:
        a, b, c, _ = self
        return Vector3.normalized(a, b, c)

    def distance_to_points(self, points: np.ndarray) -> np.ndarray:
        assert points.shape[-1] == 3
        a, b, c, d = self
        result = points[..., 0] * a + points[..., 1] * b + points[..., 2] * c + d
        return result


class Translation(np.ndarray):

    def __init__(self, input_array: np.ndarray):
        if len(input_array) != 3:
            raise ValueError("Translation must have a size of 3")
        self.__input_array = input_array


class CameraIntrinsics:

    @staticmethod
    def source_rgb_image_intrinsics():
        return CameraIntrinsics(
            1421.0684814453125,
            1421.0684814453125,
            724.485107421875,
            965.93603515625)

    @staticmethod
    def source_depth_image_intrinsics():
        return CameraIntrinsics(
            181.45,
            181.45,
            96.7,
            128.8)

    def __init__(self, fx, fy, cx, cy):
        self.__fx = fx
        self.__fy = fy
        self.__cx = cx
        self.__cy = cy

    def xyz_to_uvd(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        u = self.__fx * x / z + self.__cx
        v = self.__fy * y / z + self.__cy
        return np.stack((u, v, z), axis=-1)

    def uvd_to_xyz(self, uvd: np.ndarray) -> np.ndarray:
        u, v, d = uvd[..., 0], uvd[..., 1], uvd[..., 2]
        x = (u - self.__cx) * d / self.__fx
        y = (v - self.__cy) * d / self.__fy
        return np.stack((x, y, d), axis=-1)

    @property
    def fx(self):
        return self.__fx

    @property
    def fy(self):
        return self.__fy

    @property
    def cx(self):
        return self.__cx

    @property
    def cy(self):
        return self.__cy


class CropImagesToAspectRatio:

    @staticmethod
    def __simplify(a: int, b: int) -> Tuple[int, int]:
        gcd = np.gcd(a, b)
        return a // gcd, b // gcd

    @staticmethod
    def __to_common_denominator(a1, b1, a2, b2) -> Tuple[int, int, int, int]:
        d = np.lcm(b1, b2)
        return (d // b1) * a1, d, (d // b2) * a2, d

    @staticmethod
    def crop_images(images: np.ndarray, a, b):
        assert len(images.shape) in (2, 3, 4)
        if len(images.shape) == 2:
            images = images[..., np.newaxis]
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]

        _, h0, w0, _ = images.shape
        a0, b0 = CropImagesToAspectRatio.__simplify(h0, w0)
        a, b = CropImagesToAspectRatio.__simplify(a, b)
        print(f"Current aspect ratio: {a0, b0}")
        print(f"Current resolution: {h0, w0}")
        print(f"Target aspect ratio: {a, b}")
        h, w = None, None
        if a0 * b < a * b0:
            b0, a0, b, a = CropImagesToAspectRatio.__to_common_denominator(
                b0, a0, b, a)

            assert a0 == a

            h = h0
            w = b * h0 // a0
        else:
            a0, b0, a, b = CropImagesToAspectRatio.__to_common_denominator(
                a0, b0, a, b)

            assert b0 == b

            w = w0
            h = a * w0 // b0
        print(f"Target resolution ratio: {h, w}")

        assert h <= h0
        assert w <= w0

        h_start = (h0 - h) // 2
        w_start = (w0 - w) // 2
        return images[:, h_start:h_start + h, w_start:w_start + w]


def find_bounding_box_with_max_points_inside(points, max_size: np.ndarray):
    sx, sy, sz = max_size
    boxes = np.zeros((0, 6), dtype=np.float64)
    for k in range(8):
        begin = points - np.array([[(k & 1) * sx, (k & 2) * sy, (k & 4) * sz]])
        end = begin + np.array(max_size)
        boxes = np.concatenate((boxes, np.concatenate((begin, end), axis=1)), axis=0)
    boxes_begin = boxes[:, np.newaxis, :3]
    boxes_end = boxes[:, np.newaxis, 3:]
    points = points[np.newaxis, ...]
    is_box_has_point = np.all((points >= boxes_begin), axis=-1) & np.all((points <= boxes_end), axis=-1)
    box_points_counts = np.count_nonzero(is_box_has_point, axis=-1)
    best_box_index = np.argmax(box_points_counts)
    return boxes[best_box_index], box_points_counts[best_box_index]
