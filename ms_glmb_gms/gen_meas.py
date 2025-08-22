from gen_truth import truth
from gen_model import model
import numpy as np
from utils import read_jsonl_measurements


def gen_observation_fn(model, s, X, W):
    """
    Linear observation equation (position components only).
    Generates measurement noise if required and computes:
        Z = H*X + W
    """
    if W == 'noise':
        # generate actual measurement noise samples W
        std_noise = np.random.randn(model.D[s].shape[1], X.shape[1])    # generate a noise matrix of size (m,n) where each element is a random number drawn from std normal distribution std_noise ~ N(0, 1)
        W = np.dot(model.D[s], std_noise)   # scale the noise matrix by using the sensor observation noise std self.D[s] so that the noise matrix actually matches the sensor noise W ~ N(0,R)
        # model.D[s].shape[1] corresponds to the number of dimensions of the noise for sensor s and the number of targets/states (X.shape[1]).
    if W == 'noiseless':
        W = np.zeros(model.D[s].shape[0])
    if len(X) == 0:
        Z = []
    else:
        Z = np.dot(model.H[s], X) + W # compute observation model Z = HX + W for the sensor s. X is true position, W is measurement noise of the sensor
    return Z


class meas:
    """
    Represents a set of measurements from one or more sensors.
    Can be constructed either from ground-truth simulation
    or from reading JSONL measurement logs.
    """
    
    def __init__(self, K, Z):
        self.K = K  # number of time steps
        self.Z = Z  # dictionary: (k, s) -> np.array(z_dim, n_meas)

    @classmethod
    def from_truth(cls, model, truth):
        """
        Generate synthetic measurements from truth and model parameters.
        """
        Z = {}
        for k in range(truth.K):
            for s in range(model.N_sensors):
                Z[(k, s)] = np.empty((model.z_dim[s, 0], 0))

                if truth.N[k] > 0:
                    # Simulate detection probability
                    idx = np.nonzero(np.random.rand(truth.N[k], 1) <= model.P_D[s])[0] # detected target indices. Simulates whether target is detected by using probability P_D[s] for sensor s.
                    if len(idx) > 0:
                        Z[(k, s)] = gen_observation_fn(model, s, truth.X[k][:, idx], 'noise')

                # --- Example clutter generation (disabled for now) ---
                # LAURENS: Clutter points can be used to model unwanted data points detected by sensors caused by Environmental Interference or Multipath Effects: 
                # Audio Detektor: Non-Target Sounds, Refelctions and Reverberations, Background Noise
                # Video Detektor: objects that are not the target of localization e.g. dog, cat
                # N_c = np.random.poisson(model.lambda_c[s, 0])
                # C = np.tile(model.range_c[s][:, 0].reshape(-1, 1), (1, N_c)) + \          # number of clutter points
                #     np.diagflat(np.dot(model.range_c[s], np.array([[-1], [1]]))) @ \
                #     np.random.rand(model.z_dim[s, 0], N_c)                                # clutter generation
                # Z[(k, s)] = np.column_stack((Z[(k, s)], C))                               # measurement is union of detections and clutter

        return cls(truth.K, Z)


    @classmethod
    def from_jsonl(cls, model, truth, filepaths, sensor_idxs=None):
        """
        Build measurements from one or more JSONL files.
        Each file corresponds to one sensor index.
        Ensures that Z contains entries for all k,s combinations,
        with the same number of time steps as truth.K.
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        if sensor_idxs is None:
            sensor_idxs = list(range(len(filepaths)))

        # Read measurements from all files
        Z_raw = {}
        for filepath, s in zip(filepaths, sensor_idxs):
            Z_part = read_jsonl_measurements(filepath, model, s)
            Z_raw.update(Z_part)

        # Pre-allocate empty arrays for all k,s (same as truth.K)
        Z = {}
        for k in range(truth.K):
            for s in range(model.N_sensors):
                Z[(k, s)] = np.empty((model.z_dim[s, 0], 0))

        # Fill available entries with JSONL data
        for (k, s), val in Z_raw.items():
            if k < truth.K:  # safeguard: ignore measurements beyond truth horizon
                Z[(k, s)] = val

        return cls(truth.K, Z)


if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    meas_params = meas(model_params, truth_params)
    print(meas_params)
