import numpy as np
from gen_model import model
import json
from pathlib import Path

def gen_newstate_fn(model, Xd, V):
    """
    Linear state space equation (constant velocity (CV) model).
    X = F*Xd + V
    """
    if V == 'noise':
        V = np.linalg.multi_dot(model.sigma_V, model.B, np.randn(model.B.shape[1], len(Xd))) # process noise V
    if V == 'noiseless':
        V = np.zeros((model.B.shape[0])) # set process noise V to zero
    if len(Xd) == 0:
        X = []
    else:
        X = np.dot(model.F, Xd) + V # state space model X = FXd + V. F is the state transition matrix, Xd is the pervious state, V is the process noise 
    return X


class truth:
    """
    Models ground truth states for a set of targets across multiple time steps.
    Can be constructed either synthetically or from JSONL measurement logs.
    The class is used in tracking systems to generate true states of objects that other algorithms (like Kalman filters) are trying to estimate.
    """    
    def __init__(self, K, X, N, L, track_list, total_tracks):
        self.K = K                          # length of data/number of scans (number of time steps)
        self.X = X                          # ground truth for states of targets. A dictionary is used where each key corresponds 
                                            # to a time step (k) and the value is an array containing the true state of all targets at that time step
        self.N = N                          # ground truth number of targets for each time step
        self.L = L                          # ground truth labels of targets (k,i). ID of target ID for each time step.
        self.track_list = track_list        # absolute index target identities (for plotting) A dictionary that maps each time step (k) to a list of target indices (identifiers) that are present at that time.
        self.total_tracks = total_tracks    # total number of appearing tracks. Keeps track of the total number of targets that appear over time (across all time steps)
    
    @classmethod
    def from_synthetic(cls, model, K=100):
        """
        Generate synthetic truth data with birth/death logic.
        """
        # Pre-allocate state containers
        X = {k: np.empty((model.x_dim, 0)) for k in range(K)}
        N = {k: 0 for k in range(K)}
        L = np.zeros((K, 1))
        track_list = {k: np.empty(0, dtype=int) for k in range(K)}
        total_tracks = 0

        # target initial states and birth/death times
        nbirths = 3

        tbirth = np.zeros((nbirths), np.int32)
        tdeath = np.zeros((nbirths), np.int32)
        xstart = np.zeros((model.x_dim, nbirths))

        # create 3 targets by defining each start
        xstart[:, 0:1] = np.array([[-5], [1.0], [0], [0], [0], [0]]) # [x, vx, y, vy, z, vz]
        tbirth[0] = 1
        tdeath[0] = K + 1
        xstart[:, 1:2] = np.array([[-5], [1.0], [3], [0], [0], [0]]) # [x, vx, y, vy, z, vz]
        tbirth[1] = 1
        tdeath[1] = K + 1
        xstart[:, 2:3] = np.array([[-5], [1.0], [3], [0], [0], [0]]) # [x, vx, y, vy, z, vz]
        tbirth[2] = 1
        tdeath[2] = K + 1

        # generate the tracks
        for targetnum in range(nbirths):
            targetstate = xstart[:, targetnum]
            for k in range(tbirth[targetnum] - 1, min(tdeath[targetnum], K)):
                targetstate = gen_newstate_fn(model, targetstate, 'noiseless')
                X[k] = np.column_stack((X[k], targetstate))
                track_list[k] = np.append(track_list[k], targetnum)
                N[k] += 1
        total_tracks = nbirths

        return cls(K, X, N, L, track_list, total_tracks)

    @classmethod
    def from_jsonl(cls, model, filepath):
        """
        Read truth data from JSONL file.
        Each JSONL line contains objects with object_id and position.
        Determines K automatically from max timestamp.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"JSONL file not found: {filepath}")

        # First pass: read all lines and find max timestamp
        lines = []
        max_k = 0
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print("Malformed JSON:", line)
                    continue
                lines.append(data)
                k = round(float(data["timestamp"]) / (1 / model.T))
                if k > max_k:
                    max_k = k
        K = max_k + 1  # horizon determined from data

        # Pre-allocate state containers
        X = {k: np.empty((model.x_dim, 0)) for k in range(K)}
        N = {k: 0 for k in range(K)}
        L = np.zeros((K, 1))
        track_list = {k: np.empty(0, dtype=int) for k in range(K)}
        total_tracks = 0

        seen_object_ids = set()

        # Second pass: fill states
        for data in lines:
            k = round(float(data["timestamp"]) / (1 / model.T))
            if k >= K:
                continue

            for obj in data["objects"]:
                object_id = obj["object_id"]
                position = obj["position"]

                # Build target state [x, vx, y, vy, z, vz]
                targetstate = np.array([position[0], 0.0, position[1], 0.0, position[2], 0.0]).reshape(-1, 1)   # [x, vx, y, vy, z, vz] # TODO: Check if velocities should different from 0.0 

                X[k] = np.column_stack((X[k], targetstate))
                N[k] += 1
                track_list[k] = np.append(track_list[k], object_id - 1)  # TODO: First object_id has to be 0. WORKAROUND "object_id-1"
                L[k] = object_id

                if object_id not in seen_object_ids:
                    total_tracks += 1
                    seen_object_ids.add(object_id)

        return cls(K, X, N, L, track_list, total_tracks)

if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    print(truth_params)
