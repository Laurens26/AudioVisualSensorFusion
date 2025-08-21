import numpy as np


class model:
    def __init__(self):
        # basic parameters
        self.x_dim = 6  # 6 dimension of state vector "state space model" [x, vx, y, vy, z, vz] where representing 3D position [x,y,z] and 3D velocity [vx,vy,vz] 

        # dynamical model parameters (constant velocity (CV) model) assumes New Position = Old Position + Velocity × Time Step
        self.T = 1  # sampling period 1Hz
        self.A0 = np.array([[1, self.T],
                            [0, 1]])  # base transition matrix
        self.F = np.kron(np.eye(3, dtype='f8'), self.A0) #  state transition matrix. Kronecker product used to apply base transition matrix A0 to all three spatial dimensions (x, y, z)
        #       [ 1 1 0 0 0 0 ]
        #       [ 0 1 0 0 0 0 ]
        # F =   [ 0 0 1 1 0 0 ]
        #       [ 0 0 0 1 0 0 ]
        #       [ 0 0 0 0 1 1 ]
        #       [ 0 0 0 0 0 1 ]

        self.B0 = np.array([[(self.T ** 2) / 2], [self.T]]) # base process noise: (1) Position changes due to acceleration: 1/2*a*T**2 (2) velocity change due to acceleration: a * T
        # B0 =  [1/2]
        #       [ 1 ]
        self.B = np.concatenate((np.concatenate((self.B0, np.zeros((2, 2))), axis=1),                   # process noise. Extends base process noise B0 for all three spatial dimensions (x, y, z).
                                 np.concatenate((np.zeros((2, 1)), self.B0, np.zeros((2, 1))), axis=1),
                                 np.concatenate((np.zeros((2, 2)), self.B0), axis=1)), axis=0)
        #       [ 1/2  0   0 ]
        #       [  1   0   0 ]
        # B =   [  0  1/2  0 ]
        #       [  0   1   0 ]
        #       [  0   0  1/2]
        #       [  0   0   1 ]
        self.sigma_v = 0.25    # process noise standard deviation of the system velocity
        self.Q = self.sigma_v ** 2 * np.dot(self.B, self.B.T)  # process noise covariance

        # survival/death parameters
        self.P_S = .99
        self.Q_S = 1 - self.P_S

        # birth parameters (LMB birth model, single component only)
        self.T_birth = 4  # no. of LMB birth terms
        self.L_birth = np.zeros((self.T_birth, 1))  # no of Gaussians in each LMB birth term
        self.r_birth = np.zeros((self.T_birth, 1))  # prob of birth for each LMB birth term
        self.w_birth = np.zeros((self.T_birth, 1))  # weights of GM for each LMB birth term
        self.m_birth = np.zeros((self.T_birth, self.x_dim, 1))  # means of GM for each LMB birth term
        self.B_birth = np.zeros((self.T_birth, self.x_dim, self.x_dim))  # std of GM for each LMB birth term
        self.P_birth = np.zeros((self.T_birth, self.x_dim, self.x_dim))  # cov of GM for each LMB birth term

        self.L_birth[0] = 1  # no of Gaussians in birth term 1
        self.r_birth[0] = 0.0005  # prob of birth
        self.w_birth[0] = 1  # weight of Gaussians - must be column_vector
        self.m_birth[0, :, :] = np.array([[0], [0], [0], [0], [0], [0]])  # mean of Gaussians
        self.B_birth[0, :, :] = np.diagflat([[1], [1], [1], [1], [1], [1]])  # std of Gaussians
        self.P_birth[0, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T  # cov of Gaussians

        self.L_birth[1] = 1  # no of Gaussians in birth term 2
        self.r_birth[1] = 0.0005  # prob of birth
        self.w_birth[1] = 1  # %weight of Gaussians - must be column_vector
        self.m_birth[1, :, :] = np.array([[0], [0], [1], [0], [0], [0]])  # mean of Gaussians
        self.B_birth[1, :, :] = np.diagflat([[1], [1], [1], [1], [1], [1]])  # std of Gaussians
        self.P_birth[1, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T  # cov of Gaussians

        self.L_birth[2] = 1  # no of Gaussians in birth term 3
        self.r_birth[2] = 0.0005  # prob of birth
        self.w_birth[2] = 1  # weight of Gaussians - must be column_vector
        self.m_birth[2, :, :] = np.array([[1], [0], [0], [0], [0], [0]])  # mean of Gaussians
        self.B_birth[2, :, :] = np.diagflat([[1], [1], [1], [1], [1], [1]])  # std of Gaussians
        self.P_birth[2, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T  # cov of Gaussians

        self.L_birth[3] = 1  # no of Gaussians in birth term 4
        self.r_birth[3] = 0.0005  # prob of birth
        self.w_birth[3] = 1  # weight of Gaussians - must be column_vector
        self.m_birth[3, :, :] = [[1], [0], [1], [0], [0], [0]]  # mean of Gaussians
        self.B_birth[3, :, :] = np.diagflat([[1], [1], [1], [1], [1], [1]])  # std of Gaussians
        self.P_birth[3, :, :] = self.B_birth[0][:, :] * self.B_birth[0][:, :].T  # cov of Gaussians

        # multi-sensor observation parameters
        n_dim = 3  # noise_dim e.g. for 3D sensors [x, y, z]
        self.N_sensors = 1  # no of sensors e.g. cameras, lidar, radar
        self.z_dim = np.zeros((self.N_sensors, 1), dtype=int)  # dimensions of observation vector for each sensor. Set later e.g. z_dim = 2 for 2D camera [x, y], or z_dim = 3 for 3D Lidar [x, y, z]
        self.H = np.zeros((self.N_sensors, n_dim, self.x_dim))  # observation matrix for each sensor. Needed for vector of meaured noisy signals: z = Hx + v
        self.D = np.zeros((self.N_sensors, n_dim, n_dim))  # observation noise std deviation for each sensor
        self.R = np.zeros((self.N_sensors, n_dim, n_dim))  # observation noise covariance for each sensor. Defines how the noise in each dimension is correlated. It’s usually a diagonal matrix for independent noise components
        self.P_D = np.zeros((self.N_sensors, 1))  # probability of detection for each sensor
        self.Q_D = np.zeros((self.N_sensors, 1))  # probability of missed for each sensor
        self.lambda_c = np.zeros((self.N_sensors, 1))  # poisson clutter rate for each sensor: rate false measurements detected by the sensor, modeled as a Poisson process. e.g. 10 --> 10 clutter points per scan
        self.range_c = np.zeros((self.N_sensors, 3, 2))  # uniform clutter region for each sensor: represents the min & max region in which clutter can appear for e.g. 3D [x, y, z]
        self.pdf_c = np.zeros((self.N_sensors, 1))  # uniform clutter density for each sensor. uniform PDF is used to model the likelihood of a clutter point appearing at any given position within the clutter region.
        
        # observation parameters of sensor 1
        self.z_dim[0] = 3   # dimension of sensor measurements e.g. 3D [x, y, z]
        self.H[0] = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]) # 1st row extracts x, 2nd row extracts y, 3rd row extracts z from state vector x: [x, vx, y, vy, z, vz] 
                                                                                           # by computing the first part of vector of meaured noisy signals: "z = Hx" + v
        self.D[0] = np.diag([0.1, 0.1, 0.1])                                           # observation noise std deviation. Diagonal matrix as x, y, z have independend noise std deviation 
        self.R[0] = self.D[0] * self.D[0].T                                         # observation noise covariance: R = DD^T. R describes the statistical properties of the noise
        self.P_D[0] = .5                                                           # probability of detection = 0.95 
        self.Q_D[0] = 1 - self.P_D[0]                                               # probability of missed = 0.05 
        self.lambda_c[0] = 1                                                       # 10 clutter points per scan
        self.range_c[0] = np.array([[-20, 20], [-20, 20], [-20, 20]])   # uniform clutter region for each dimension x, y, z is [-1000, 1000] 
        self.pdf_c[0] = 1 / np.prod(self.range_c[0][:, 1] - self.range_c[0][:, 0])  # calculates the uniform clutter density based on the volume of the clutter region e.g. 1 / 8,000,000,000

        # # observation parameters of sensor 2
        # self.z_dim[1] = 3
        # self.H[1] = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
        # self.D[1] = np.diag([0.1, 0.1, 0.1])
        # self.R[1] = self.D[1] * self.D[1].T
        # self.P_D[1] = .95
        # self.Q_D[1] = 1 - self.P_D[1]
        # self.lambda_c[1] = 10
        # self.range_c[1] = np.array([[-20, 20], [-20, 20], [-20, 20]])
        # self.pdf_c[1] = 1 / np.prod(self.range_c[1][:, 1] - self.range_c[1][:, 0])

        # # # observation parameters of sensor 3
        # self.z_dim[2] = 3
        # self.H[2] = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
        # self.D[2] = np.diag([0.1, 0.1, 0.1])
        # self.R[2] = self.D[2] * self.D[2].T
        # self.P_D[2] = .95
        # self.Q_D[2] = 1 - self.P_D[2]
        # self.lambda_c[2] = 10
        # self.range_c[2] = np.array([[-20, 20], [-20, 20], [-20, 20]])
        # self.pdf_c[2] = 1 / np.prod(self.range_c[2][:, 1] - self.range_c[2][:, 0])


if __name__ == '__main__':
    model_params = model()
    print(model_params)
