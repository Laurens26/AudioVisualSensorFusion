from gen_truth import truth
from gen_model import model
import numpy as np
import json
import time


def gen_observation_fn(model, s, X, W):
    # linear observation equation (position components only)
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
    def __init__(self, model, truth, generate_measurements=False):
        # variables
        self.K = truth.K
        self.Z = {}
        for k in range(0, truth.K):
            for s in range(model.N_sensors):
                self.Z[(k, s)] = np.empty((model.z_dim[s, 0], 0))

        if generate_measurements is True:
            # generate measurements
            for k in range(0, truth.K):
                for s in range(model.N_sensors):
                    if truth.N[k] > 0:  #  Checks if there are any targets present at time step k
                        idx = np.nonzero(np.random.rand(truth.N[k], 1) <= model.P_D[s])[0]  # detected target indices. Simulates whether target is detected by using probability P_D[s] for sensor s.
                        self.Z[k, s] = gen_observation_fn(model, s, truth.X[k][:, idx], 'noise')  # single target observations if detected

                        # LAURENS: Clutter points can be used to model unwanted data points detected by sensors caused by Environmental Interference or Multipath Effects: 
                        # Audio Detektor: Non-Target Sounds, Refelctions and Reverberations, Background Noise
                        # Video Detektor: objects that are not the target of localization e.g. dog, cat
                        N_c = np.random.poisson(model.lambda_c[s, 0])  # number of clutter points
                        C = np.tile(model.range_c[s][:, 0].reshape(-1, 1), (1, N_c)) + \
                            np.diagflat(np.dot(model.range_c[s], np.array([[-1], [1]]))) @ \
                            np.random.rand(model.z_dim[s, 0], N_c)  # clutter generation
                        self.Z[k, s] = np.column_stack((self.Z[k, s], C))  # measurement is union of detections and clutter
                #
            #
    
    def run_jsonl_main_loop(self, jsonl_filepath, dim=3, sensor_idx=0):
            jsonl = []  # List of JSON dictionaries updated by jsonl reader
            processed_lines = 0
            file_position = 0  # Byte offset position in file

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

                            if dim == 2:
                                position = obj["position"]
                                position[2] = 0
                                # position = np.array([obj["position"][0], obj["position"][1]])
                            else:
                                position = obj["position"]

                            # Check if in measurement area
                            AREA_X_MIN = 0.0
                            AREA_X_MAX = 4.66
                            AREA_Y_MIN = 0.0
                            AREA_Y_MAX = 4.66
                            AREA_Z_MIN = 0.0
                            AREA_Z_MAX = 3.0


                            if (AREA_X_MIN <= position[0] <= AREA_X_MAX and
                                AREA_Y_MIN <= position[1] <= AREA_Y_MAX 
                                # and AREA_Z_MIN <= position[2] <= AREA_Z_MAX
                                ):   
                                # Get timestamp integer k
                                k = int(relative_timestamp*10)  # Use the timestamp as the index # TODO: Use different time resolution than WORKAROUND "*10"
                                # Stack new observation to existing ones to allow for multiple measurements for the same timestamp and sensor
                                if dim == 2:
                                    # entry = np.hstack((self.Z[k, sensor_idx], np.array(position).reshape(2, 1)))
                                    entry = np.hstack((self.Z[k, sensor_idx], np.array(position).reshape(3, 1)))
                                else:
                                    entry = np.hstack((self.Z[k, sensor_idx], np.array(position).reshape(3, 1)))
                                self.Z[k, sensor_idx] = entry
                                # self.Z[k, sensor_idx] = np.hstack((self.Z[k, sensor_idx], np.array([position[0], position[1], 0]).reshape(3, 1))) # 2D TESTING ONLY
                                
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
    meas_params = meas(model_params, truth_params)
    print(meas_params)
