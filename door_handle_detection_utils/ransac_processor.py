from door_handle_detection_utils.data_processor import DataProcessor
import numpy as np


class RANSACProcessor(DataProcessor):

    def __init__(self, random_samples_number, score_points_number):
        self.random_samples_number = random_samples_number
        self.score_points_number = score_points_number

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[0]
        m = 3*self.random_samples_number

        assert m <= n

        points = data[np.random.choice(n, m)].reshape(-1, 3, 3)

        n1 = points[:, 1, :] - points[:, 0, :]
        n2 = points[:, 2, :] - points[:, 0, :]

        a = n1[:, 1] * n2[:, 2] - n1[:, 2] * n2[:, 1]
        b = n1[:, 2] * n2[:, 0] - n1[:, 0] * n2[:, 2]
        c = n1[:, 0] * n2[:, 1] - n1[:, 1] * n2[:, 0]
        d = -a * points[:, 0, 0] - b * points[:, 0, 1] - c * points[:, 0, 2]

        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        c = c.reshape(-1, 1)
        d = d.reshape(-1, 1)

        planes = np.hstack((a, b, c, d))

        assert planes.shape == (self.random_samples_number, 4)

        planes = planes / ((planes[:, :-1]**2).sum(axis=1)**0.5).reshape(-1, 1)

        score_points = data[np.random.choice(n, self.score_points_number)]
        score_points = np.concatenate((score_points.reshape(-1, 1, 3), np.ones((self.score_points_number, 1, 1))), axis=2)

        scores = np.abs((planes * score_points).sum(axis=-1)).mean(axis=0)

        result = planes[np.nanargmin(scores)]

        return result
