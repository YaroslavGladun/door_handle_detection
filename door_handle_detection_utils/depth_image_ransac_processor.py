import numpy as np

from door_handle_detection_utils.ransac_processor import RANSACProcessor


class DepthImageRANSACProcessor(RANSACProcessor):

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        assert len(data.shape) == 2

        n = data.shape[0] * data.shape[1]
        m = 3 * self.random_samples_number

        assert m <= n

        indexes = np.random.choice(n, m).reshape(-1, 1)
        indexes = np.concatenate((indexes // data.shape[1], indexes % data.shape[1]), axis=1)
        points = np.concatenate((indexes, data[indexes[:, 0], indexes[:, 1]].reshape(-1, 1)), axis=1)
        points = points.reshape(-1, 3, 3)

        score_indexes = np.random.choice(n, self.score_points_number).reshape(-1, 1)
        score_indexes = np.concatenate((score_indexes // data.shape[1], score_indexes % data.shape[1]), axis=1)
        score_points = np.concatenate(
            (score_indexes, data[score_indexes[:, 0], score_indexes[:, 1]].reshape(-1, 1)), axis=1)
        score_points = np.concatenate(
            (score_points.reshape(-1, 1, 3), np.ones((self.score_points_number, 1, 1))), axis=2)

        return self.find_nearest_plane(points, score_points)
