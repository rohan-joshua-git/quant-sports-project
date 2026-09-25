import numpy as np


class BallKalman:
    """Constant-velocity Kalman filter for the ball centre, in pixels.

    State is [x, y, vx, vy], with velocity in pixels per frame. Forward-only: every
    estimate uses frames up to and including the current one, never later ones, so
    it is point-in-time correct and works on a live stream.
    """

    def __init__(self, accel_std=4.0, meas_std=2.0, init_vel_std=15.0):
        # Position moves by velocity each frame; velocity stays the same.
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        # Only the position is observed.
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        # Process noise: unknown acceleration (kicks, bounces, camera pans) of
        # accel_std px/frame^2, which moves position by a/2 and velocity by a.
        g = np.array([[0.5, 0.0], [0.0, 0.5], [1.0, 0.0], [0.0, 1.0]])
        self.Q = accel_std ** 2 * g @ g.T
        self.R = meas_std ** 2 * np.eye(2)
        self.init_vel_std = init_vel_std
        self.meas_std = meas_std
        self.reset()

    def reset(self):
        self.x = None
        self.P = None

    @property
    def initialized(self):
        return self.x is not None

    def initiate(self, z):
        """Start from a first detection: position known, velocity unknown."""
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.meas_std ** 2, self.meas_std ** 2,
                          self.init_vel_std ** 2, self.init_vel_std ** 2])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def distance_sq(self, z):
        """Squared Mahalanobis distance of a detection from the prediction.

        Distance measured in standard deviations of the predicted position, so the
        allowed distance grows automatically while the ball goes unseen.
        """
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y @ np.linalg.solve(S, y))

    def update(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[:2]

    @property
    def position_std(self):
        """Uncertainty of the position estimate, in pixels (average of x and y)."""
        return float(np.sqrt((self.P[0, 0] + self.P[1, 1]) / 2))
