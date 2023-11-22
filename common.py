import numpy as np
from typing import Tuple


class Point(np.ndarray):

    @staticmethod
    def normalized(a, b, c):
        p = Point([a, b, c])
        return p / np.linalg.norm(p)

    def __new__(cls, input_array):
        if len(input_array) != 3:
            raise ValueError("Point must have a size of 3")

        obj = np.asarray(input_array).view(cls)
        return obj


class Plane(np.ndarray):

    @staticmethod
    def normalized(a, b, c, d):
        abc_norm = np.linalg.norm([a, b, c])
        a, b, c, d = a / abc_norm, b / abc_norm, c / abc_norm, d / abc_norm
        return Plane([a, b, c, d])

    def __new__(cls, input_array):
        if len(input_array) != 4:
            raise ValueError("Plane must have a size of 4")

        obj = np.asarray(input_array).view(cls)
        return obj

    def norm(self) -> Point:
        a, b, c, _ = self
        return Point.normalized(a, b, c)


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