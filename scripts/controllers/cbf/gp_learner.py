import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from collections import deque
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ResidualDynamicsGP:
    """
    Learns the residual dynamics d(x) = v_dot_actual - v_dot_nominal using a GP.
    Inputs: z = [v, u] (velocity, command)
    Outputs: d (residual acceleration vector)
    """

    def __init__(self, buffer_size=1000, length_scale=1.0, noise_level=0.1):
        self.buffer_size = buffer_size

        # Circular buffer for (z, d) pairs
        self.data_buffer_z = deque(maxlen=buffer_size)
        self.data_buffer_d = deque(maxlen=buffer_size)

        # Kernel: RBF for smoothness + WhiteKernel for noise
        # Note: length_scale effectively scales the RBF bandwidth
        kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)

        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=0, normalize_y=True
        )
        self.is_trained = False

    def add_data(self, z, d):
        """
        z: np.array of shape (input_dim,) -> e.g. [vx, vy, vz, ux, uy, uz]
        d: np.array of shape (output_dim,) -> e.g. [dx, dy, dz]
        """
        self.data_buffer_z.append(z)
        self.data_buffer_d.append(d)

    def train(self):
        """
        Fit the GP to the current buffer.
        Returns True if training successful (enough data), False otherwise.
        """
        if len(self.data_buffer_z) < 10:
            return False

        X = np.array(self.data_buffer_z)
        y = np.array(self.data_buffer_d)

        self.gp.fit(X, y)
        self.is_trained = True
        return True

    def predict(self, z):
        """
        Returns mean (mu) and standard deviation (sigma) for the residual at z.
        z: shape (input_dim,)
        """
        if not self.is_trained:
            return np.zeros(3), np.zeros(1)

        # Reshape for sklearn (1, n_features)
        z_in = np.atleast_2d(z)
        mu, sigma = self.gp.predict(z_in, return_std=True)

        return mu.flatten(), sigma.flatten()
