import numpy as np
from gen_model import model
import json
import time

def gen_newstate_fn(model, Xd, V):
    # linear state space equation (constant velocity (CV) model)
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
    The truth class is designed to model ground truth states for a set of targets across multiple time steps. 
    The class is used in tracking systems to generate true states of objects that other algorithms (like Kalman filters) are trying to estimate.
    """
    def __init__(self, model, generate_truth=False):
        # define variables
        self.K = 41  # length of data/number of scans (number of time steps)
        self.X = {key: np.empty((model.x_dim, 0)) for key in range(0, self.K)}  # ground truth for states of targets. A dictionary is used where each key corresponds 
                                                                                # to a time step (k) and the value is an array containing the true state of all targets at that time step
        self.N = {key: 0 for key in range(0, self.K)}  # ground truth number of targets for each time step
        self.L = np.zeros((self.K, 1))  # ground truth labels of targets (k,i). ID of target ID for each time step.
        self.track_list = {key: np.empty(0, dtype=int) for key in
                           range(0, self.K)}  # absolute index target identities (for plotting) A dictionary that maps each time step (k) to a list of target indices (identifiers) that are present at that time.
        self.total_tracks = 0  # total number of appearing tracks. Keeps track of the total number of targets that appear over time (across all time steps)

        if generate_truth is True:
            # target initial states and birth/death times
            nbirths = 3

            tbirth = np.zeros((nbirths), np.int32)
            tdeath = np.zeros((nbirths), np.int32)
            xstart = np.zeros((model.x_dim, nbirths))

            # create 12 targets by defining each start
            xstart[:, 0:1] = np.array([[0], [0.1], [3.33], [0], [0], [0]]) # [x, vx, y, vy, z, vz]
            tbirth[0] = 1
            tdeath[0] = self.K + 1
            xstart[:, 1:2] = np.array([[0], [0.1], [2.33], [0], [0], [0]]) # [x, vx, y, vy, z, vz]
            tbirth[1] = 1
            tdeath[1] = self.K + 1
            xstart[:, 2:3] = np.array([[0], [0.1], [1.33], [0], [0], [0]])
            tbirth[2] = 1
            tdeath[2] = self.K + 1

            # xstart[:, 3:4] = np.array([[400], [-7], [-600], [-4], [200], [-3]])
            # tbirth[3] = 20
            # tdeath[3] = self.K + 1
            # xstart[:, 4:5] = np.array([[400], [-2.5], [-600], [10], [200], [0]])
            # tbirth[4] = 20
            # tdeath[4] = self.K + 1
            # xstart[:, 5:6] = np.array([[0], [7.5], [0], [-5], [0], [5]])
            # tbirth[5] = 20
            # tdeath[5] = self.K + 1

            # xstart[:, 6:7] = np.array([[-800], [12], [-200], [7], [-400], [3]])
            # tbirth[6] = 40
            # tdeath[6] = self.K + 1
            # xstart[:, 7:8] = np.array([[-200], [15], [800], [-10], [600], [-10]])
            # tbirth[7] = 40
            # tdeath[7] = self.K + 1

            # xstart[:, 8:9] = np.array([[-800], [3], [-200], [15], [-400], [5]])
            # tbirth[8] = 60
            # tdeath[8] = self.K + 1
            # xstart[:, 9:10] = np.array([[-200], [-3], [800], [-15], [600], [-10]])
            # tbirth[9] = 60
            # tdeath[9] = self.K + 1

            # xstart[:, 10:11] = np.array([[0], [-20], [0], [-15], [0], [-15]])
            # tbirth[10] = 80
            # tdeath[10] = self.K + 1
            # xstart[:, 11:12] = np.array([[-200], [15], [800], [-5], [600], [-7]])
            # tbirth[11] = 80
            # tdeath[11] = self.K + 1

            # generate the tracks
            for targetnum in range(nbirths):
                targetstate = xstart[:, targetnum]
                for k in range(tbirth[targetnum] - 1, min(tdeath[targetnum], self.K)):
                    targetstate = gen_newstate_fn(model, targetstate, 'noiseless')
                    self.X[k] = np.column_stack((self.X[k], targetstate))
                    self.track_list[k] = np.append(self.track_list[k], targetnum)
                    self.N[k] = self.N[k] + 1
            self.total_tracks = nbirths

    
    def run_jsonl_main_loop(self, jsonl_filepath, dim=3,):
            jsonl = []  # List of JSON dictionaries updated by jsonl reader
            processed_lines = 0
            file_position = 0  # Byte offset position in file

            # Assuming 'seen_object_ids' is a set that stores the object IDs you've already encountered
            seen_object_ids = set()

            while True:
                try:
                    with open(jsonl_filepath, 'r') as f:
                        # Set the reading position
                        f.seek(file_position) # Seek to last read position
                        new_lines = f.readlines()
                        file_position = f.tell() # Update the current file position

                    if not new_lines:
                        time.sleep(1.0) # Avoid busy waiting
                        continue

                    # Parse and enrich new JSON lines
                    for line in new_lines:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            jsonl.append(data)
                        except json.JSONDecodeError:
                            print("Malformed JSON:", line)

                    # Create batch for further processing
                    jsonl_batch = jsonl[-len(new_lines):]
                    
                    # Extract trajectories per object
                    for line in jsonl_batch:
                        relative_timestamp = float(line["timestamp"])

                        for obj in line["objects"]:
                            object_id = obj["object_id"]

                            if dim == 2:
                                position = obj["position"]
                                position[2] = 0
                                # position = np.array([obj["position"][0], obj["position"][1]])
                            else:
                                position = obj["position"]


                            # Get timestamp integer k
                            k = int(relative_timestamp*10)  # Use the timestamp as the index # TODO: Use different time resolution than WORKAROUND "*10"

                            # Append position to X for the respective timestamp
                            if dim == 2:
                                # targetstate = [position[0], 0.0, position[1], 0.0]    # [x, vx, y, vy, z, vz] # TODO: Check if velocities should different from 0.0 
                                targetstate = [position[0], 0.0, position[1], 0.0, position[2], 0.0]    # [x, vx, y, vy, z, vz] # TODO: Check if velocities should different from 0.0 
                            else:
                                targetstate = [position[0], 0.0, position[1], 0.0, position[2], 0.0]    # [x, vx, y, vy, z, vz] # TODO: Check if velocities should different from 0.0 

                            self.X[k] = np.column_stack((self.X[k], targetstate))

                            # Update the number of targets at the current timestamp
                            self.N[k] += 1

                            # Update the target's track list at the current timestamp
                            self.track_list[k] = np.append(self.track_list[k], object_id-1) # TODO: First object_id has to be 0. WORKAROUND "object_id-1"

                            # Update the label for the target (track ID) for the current timestamp
                            self.L[k] = object_id

                            # Increment the total track count
                            # Check if this object has been seen before
                            if object_id not in seen_object_ids:
                                # It's a new "track birth", so increment the total_tracks count
                                self.total_tracks += 1
                                # Add this object_id to the set to keep track of it
                                seen_object_ids.add(object_id)

                    # Update count of processed lines
                    processed_lines += len(new_lines)
                    print("---")

                    # TODO: Only for now, return after first JSONL reading iteration
                    return

                except FileNotFoundError:
                    print("File not found, retrying...")
                    time.sleep(1.0)

                except Exception as e:
                    print("Unexpected error:", e)
                    time.sleep(1.0)

if __name__ == '__main__':
    model_params = model()
    truth_params = truth(model_params)
    print(truth_params)
