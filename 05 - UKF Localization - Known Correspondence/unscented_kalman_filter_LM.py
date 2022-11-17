'''
Created on Feb 28, 2020
@author: Oyindamola Omotuyi
Unscented Kalman Filter Localization with Known Correspondence Algorithm for a differential drive robot
Robot: Turtlebot3 robot in ROS/Gazebo
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import multi_dot
from numpy import linalg as nl
import math
from scipy.linalg import block_diag
import scipy

import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.loc'] = 'upper left'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'


#Set Data file variables
DATA_DIR = "./data/"
true_data = DATA_DIR+'true_data.csv'
sensor_data = DATA_DIR+'sensor_data.csv'

class UKFLocalization:
    def __init__(self):

        self.generate_data()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.state = np.array([[self.x], [self.y], [self.theta]]) #pos, vel, acc

        self.prev_time = 0

        self.estimateCovariance = 0.1**2*np.identity(3) #P


        self.processNoiseCovariance = np.array([[.1**2,0],
                                                [0, .1**2]], dtype="float64") #Q for the control inputs



        self.observationCovariance = np.array([[.2**2, 0],
                                               [0, .1**2]], dtype="float64") #R

        #initial guess
        self.linear_vel = 0.0001
        self.angular_vel = 0.000001


        self.STATE_SIZE = 3
        self.INPUT_SIZE = 2
        self.MEASUREMENT_SIZE = 2
        self.TOTAL_SIZE = self.STATE_SIZE + self.INPUT_SIZE + self.MEASUREMENT_SIZE



        ##LANDMARKS
        self.landmark_size = 4
        self.landmark_id = 0
        self.map = np.empty((self.landmark_size,3))

        #UKF PARAMETERS
        self.augmented_state = np.row_stack((self.state, np.zeros((4,1))))
        self.augmented_covariance = block_diag(self.estimateCovariance, self.processNoiseCovariance, self.observationCovariance)

        self.alpha = .04
        self.beta = 2
        self.k = 0
        self.lambd = self.alpha**2*(self.TOTAL_SIZE + self.k) -self.TOTAL_SIZE

        self.n_sigma = 2*(self.STATE_SIZE + self.INPUT_SIZE + self.MEASUREMENT_SIZE) + 1
        self.weights_mean = np.empty((1, self.n_sigma))
        self.weights_covariance = np.zeros(self.n_sigma)
        self.generateWeights()


    def generate_data(self):
        self.true_data= np.genfromtxt(true_data, delimiter=',')
        self.sensor_data= np.genfromtxt(sensor_data, delimiter=',')


        self.true_data = np.array(self.true_data[1:], dtype=np.float64)
        self.sensor_data = np.array(self.sensor_data[1:], dtype=np.float64)

    def generateWeights(self):
        self.weights_mean[0,0] = self.lambd / (self.TOTAL_SIZE + self.lambd)

        self.weights_covariance[0] = (self.lambd / (self.TOTAL_SIZE + self.lambd))+(1- self.alpha**2 + self.beta)


        self.weights_mean[:,1:] = 1/ (2*(self.TOTAL_SIZE + self.lambd))
        self.weights_covariance[1:] = 1/ (2*(self.TOTAL_SIZE + self.lambd))


    def generateSigmaPoints(self):
        sigma_points = np.zeros((self.n_sigma, self.TOTAL_SIZE))

        cov_sig = scipy.linalg.sqrtm((self.TOTAL_SIZE+self.lambd)*(self.augmented_covariance))

        sigma_points[0] = self.augmented_state.T
        for i in range(self.TOTAL_SIZE):
            sigma_points[i+1] = self.augmented_state.T + cov_sig[i]
            sigma_points[i+1+self.TOTAL_SIZE] = self.augmented_state.T - cov_sig[i]

        return sigma_points.T


#
    def wrapTheta(self):
        if(self.theta >  2*np.pi):
            print("angle", self.theta)
            self.theta -= 2*np.pi
            print("final", self.theta)


    def separateSigmas(self, sigma_points):
        sig_x = sigma_points[0:self.STATE_SIZE,:]
        sig_u = sigma_points[self.STATE_SIZE:self.STATE_SIZE+self.INPUT_SIZE,:]
        sig_z = sigma_points[self.STATE_SIZE+self.INPUT_SIZE:self.TOTAL_SIZE,:]
        return sig_x, sig_u, sig_z

    def dynamics(self, sigma_points, delta):
        sigma_dynamics = np.empty((self.STATE_SIZE, self.n_sigma))
        self.sig_x, self.sig_u, self.sig_z = self.separateSigmas(sigma_points)

        self.linear_vel_sigmas = self.linear_vel + self.sig_u[0]
        self.angular_vel_sigmas = self.angular_vel + self.sig_u[1]

        self.theta_sigmas = self.sig_x[2]


        self.v_w = np.divide(self.linear_vel_sigmas, self.angular_vel_sigmas, dtype="float64")

        sigma_dynamics[0] =self.sig_x[0] - self.v_w*np.sin(self.theta_sigmas) + self.v_w*np.sin(self.theta_sigmas + self.angular_vel_sigmas*delta)
        sigma_dynamics[1] = self.sig_x[1] + self.v_w*np.cos(self.theta_sigmas) - self.v_w*np.cos(self.theta_sigmas + self.angular_vel_sigmas*delta)
        sigma_dynamics[2] = self.sig_x[2] + self.angular_vel_sigmas * delta

        return sigma_dynamics


    def computeMean(self, y):
        return np.sum(self.weights_mean*y, axis=1, dtype="float64").reshape(-1,1)

    def computeCovariance(self, n1, n2, n3, n4, size):
        covariance = np.zeros(size)
        for i in range(self.n_sigma):
            diff = (n1[:,i] - n2.T).T
            diff_T = (n3[:,i] - n4.T)
            covariance += self.weights_covariance[i] * np.dot(diff,diff_T)
        return covariance

    def predict(self, delta):
        self.augmented_state = np.row_stack((self.state, np.zeros((4,1))))

        alpha_1 = 0.3;
        alpha_2 = 0.05;
        alpha_3 = 0.05;
        alpha_4 = 0.3;

        M_sig_1 = alpha_1*self.linear_vel**2 + alpha_2*self.angular_vel**2
        M_sig_2 = alpha_3*self.linear_vel**2 + alpha_4*self.angular_vel**2

        self.processNoiseCovariance = np.array([[M_sig_1,0],
                                                [0, M_sig_2]],  dtype="float64") #Q for the control inputs

        self.augmented_covariance = block_diag(self.estimateCovariance, self.processNoiseCovariance, self.observationCovariance)

        self.sigmas = self.generateSigmaPoints()


        self.X_x = self.dynamics(self.sigmas, delta)
        self.state = self.computeMean(self.X_x)

        self.estimateCovariance = self.computeCovariance(self.X_x, self.state, self.X_x, self.state, (3,3))

        self.wrapTheta()

        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]

        self.sig_x, self.sig_u, self.sig_z1 = self.separateSigmas(self.sigmas)



    def update(self):
        self.augmented_state = np.row_stack((self.state, np.zeros((4,1))))

        self.augmented_covariance = block_diag(self.estimateCovariance, self.processNoiseCovariance, self.observationCovariance)

        self.sigmas = self.generateSigmaPoints()


        self.sig_x, self.sig_u, self.sig_z = self.separateSigmas(self.sigmas)

        self.landmark = self.map[self.landmark_id]
        self.landmark_pos = np.array([self.landmark[0], self.landmark[1]])

        self.landmark_signature = self.landmark[2]



        self.Z = np.zeros((self.MEASUREMENT_SIZE, self.n_sigma))
        self.Z[0] = np.sqrt((self.landmark_pos[0] - self.sig_x[0]) **2 + (self.landmark_pos[1] - self.sig_x[1]) **2) + self.sig_z1[0]
        self.Z[1] = np.arctan2(self.landmark_pos[1]-self.sig_x[1], self.landmark_pos[0] - self.sig_x[0])-self.sig_x[2] + self.sig_z1[1]


        self.z = self.computeMean(self.Z)

        self.innovation_covariance = (self.weights_covariance.reshape(1,15)*(self.Z - self.z))@((self.Z - self.z).T)
        self.sigma_t = (self.weights_covariance.reshape(1,15)*(self.sig_x - self.state))@((self.Z - self.z).T)

#        self.innovation_covariance = self.computeCovariance(self.Z, self.z, self.Z, self.z,(2,2))
#        self.sigma_t = self.computeCovariance(self.sig_x , self.state, self.Z, self.z, (3,2))
#
        #Update Mean and Covariance
        self.kalman_gain = np.dot(self.sigma_t, inv(self.innovation_covariance))
        self.state = self.state + np.dot(self.kalman_gain, (self.measurement - self.z))


        self.wrapTheta()
        self.estimateCovariance = self.estimateCovariance - multi_dot([self.kalman_gain, self.innovation_covariance, self.kalman_gain.T])
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]

if __name__ == '__main__':

    filter_ = UKFLocalization()

    estimated_state = np.empty((filter_.true_data.shape[0], 3))

    sensor_data = filter_.sensor_data
    true_data = filter_.true_data
    landmarks = np.array([[-1.5,2.5], [-1.5,0], [1.5,0], [1.5,2.5]])
    filter_.landmark_size = landmarks.shape[0]

    filter_.map = np.zeros((filter_.landmark_size,3))
    measurements = np.zeros((filter_.true_data.shape[0], filter_.landmark_size, 2))

    landmarks_range = np.column_stack((filter_.sensor_data[:,1], filter_.sensor_data[:,3], filter_.sensor_data[:,5], filter_.sensor_data[:,7]))

    landmarks_bearing = np.column_stack((filter_.sensor_data[:,2], filter_.sensor_data[:,4], filter_.sensor_data[:,6], filter_.sensor_data[:,8]))

    ##DATA GENERATION
    for j in range(len(filter_.true_data)):
        for i in range(len(landmarks)):
            filter_.map[i] = np.array([landmarks[i,0], landmarks[i,1], i])

            measurements[j][i] = np.array([landmarks_range[j][i], landmarks_bearing[j][i]])



    #ALGORITHM
    for j in range(len(filter_.true_data)):
        current_time = filter_.sensor_data[j][0]
        delta = current_time - filter_.prev_time
        filter_.prev_time = current_time

        filter_.linear_vel = filter_.true_data[j][4]
        filter_.angular_vel = filter_.true_data[j][5]

        filter_.predict(delta)


        for i in range(filter_.landmark_size):

            if(~np.isnan(measurements[j][i][0])):
                filter_.landmark_id = i
                filter_.measurement = np.array([measurements[j][i]]).T

                filter_.update()


        estimated_state[j] =filter_.state.T




    time = filter_.true_data[:][:,0]
    true_x = filter_.true_data[:][:,1]
    true_y = filter_.true_data[:][:,2]
    true_theta = filter_.true_data[:][:,3]

    est_x = estimated_state[:][:,0]
    est_y = estimated_state[:][:,1]
    est_theta = estimated_state[:][:,2]


#
end_t= len(est_x) -1

#------------------------- PLOTS ---------------------#
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('UKF Localization with Range and Bearing Measurements')

#x and t
ax1.plot(time, true_x, '--r', label="True")

ax1.plot(time[0], true_x[0], 'ro', markersize=8.5, label= 'Start')
ax1.plot(time[end_t], true_x[end_t], 'bo', markersize=8.5, label ='End')
ax1.plot(time, est_x, '--b', label="Estimated")
ax1.plot(time[0], est_x[0], 'ro', markersize=8.5, label= 'Start')
ax1.plot(time[end_t], est_x[end_t], 'bo', markersize=8.5, label ='End')
ax1.grid(True)
ax1.set(xlabel='Time (secs)', ylabel="X Position (m)")
ax1.set_title('Estimated X Position vs True X Position')
ax1.legend()

#y and t
ax2.plot(time, true_y, '--r', label="True")
ax2.plot(time[0], true_y[0], 'ro', markersize=8.5, label= 'Start')
ax2.plot(time[end_t], true_y[end_t], 'bo', markersize=8.5, label ='End')
ax2.plot(time, est_y, '--b', label="Estimated")
ax2.plot(time[0], est_y[0], 'ro', markersize=8.5, label= 'Start')
ax2.plot(time[end_t], est_y[end_t], 'bo', markersize=8.5, label ='End')
ax2.grid(True)
ax2.set(xlabel='Time (secs)', ylabel="Y Position (m)")
ax2.set_title('Estimated Y Position vs True Y Position')
ax2.legend()

#theta and t
ax3.plot(time, true_theta, '--r', label="True")
ax3.plot(time[0], true_theta[0], 'ro', markersize=8.5, label= 'Start')
ax3.plot(time[end_t], true_theta[end_t], 'bo', markersize=8.5, label ='End')
ax3.plot(time, est_theta, '--b', label="Estimated")
ax3.plot(time[0], est_theta[0], 'ro', markersize=8.5, label= 'Start')
ax3.plot(time[end_t], est_theta[end_t], 'bo', markersize=8.5, label ='End')
ax3.grid(True)
ax3.set(xlabel='Time (secs)', ylabel=r"$\bf{\theta}$ (rad)")
ax3.set_title('Estimated Heading vs True Heading')
ax3.legend()

#X and Y
ax4.plot(true_x, true_y, '--r', label="True")
ax4.plot(true_x[0], true_y[0], 'ro', markersize=8.5, label= 'Start')
ax4.plot(true_x[end_t], true_y[end_t], 'bo', markersize=8.5, label ='End')
ax4.plot(est_x, est_y, '--b', label="Estimated")
ax4.plot(est_x[0], est_y[0], 'ro', markersize=8.5, label= 'Start')
ax4.plot(est_x[end_t], est_y[end_t], 'bo', markersize=8.5, label ='End')
ax4.grid(True)
ax4.set(xlabel='X (m)', ylabel="Y (m)")
ax4.set_title('Estimated Robot Trajectory vs True Robot Trajectory')

ax4.plot(landmarks[0,0],landmarks[0,1], 'k*', markersize=10, label='landmarks')
for i in range(1,len(landmarks)):
    ax4.plot(landmarks[i,0],landmarks[i,1], 'k*', markersize=10)
ax4.legend()
plt.show()
