'''
Created on Feb 14, 2020
@author: Oyindamola Omotuyi
Unscented Kalman Filter Algorithm for a differential drive robot
Robot: Turtlebot3 robot in ROS/Gazebo
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import multi_dot
import scipy.linalg

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


class UKF:
    def __init__(self):

        self.generate_data()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.state = np.array([[self.x], [self.y], [self.theta]]) #pos, vel, acc
        self.prev_time = 0

        self.estimateCovariance =0.* np.identity(3) #P

        self.processNoiseCovariance = np.array([[0.1**2, 0, 0],
                                                [0, 0.1**2, 0],
                                                [0, 0, 0.1**2]], dtype="float64") #Q for the control inputs

        self.observationMatrix = np.array([[1, 0, 0],
                                           [0, 1, 0]]) #H

        self.observationCovariance = np.array([[.3**2, 0],
                                               [0, .3**2]], dtype="float64") #R


        self.linear_vel = 0
        self.angular_vel = 0
        self.STATE_SIZE = 3
        self.count = 0

        #UKF PARAMETERS

        self.alpha = .04
        self.beta = 2
        self.k = 0
        self.lambd = self.alpha**2*(self.STATE_SIZE + self.k) -self.STATE_SIZE

        self.n_sigma = 2*self.STATE_SIZE + 1
        self.weights_mean = np.empty((1, self.n_sigma))
        self.weights_covariance = np.zeros(self.n_sigma)
        self.generateWeights()


    def setInitState(self):
        self.state = np.array([[self.x], [self.y], [self.theta]])

    def setInitState_(self, x, y, theta):
        self.state = np.array([[x], [y], [theta]])


    def wrapTheta(self):
        if(self.state[2] >  2*np.pi):
            self.state[2] -= 2*np.pi
        elif(self.state[2] <  -2*np.pi):
            self.state[2] += 2*np.pi


    def generate_data(self):
        self.true_data= np.genfromtxt(true_data, delimiter=',')
        self.sensor_data= np.genfromtxt(sensor_data, delimiter=',')

        self.true_data = np.array(self.true_data[1:], dtype=np.float64)
        self.sensor_data = np.array(self.sensor_data[1:], dtype=np.float64)


    def generateWeights(self):
        self.weights_mean[0,0] = self.lambd / (self.STATE_SIZE + self.lambd)

        self.weights_covariance[0] = (self.lambd / (self.STATE_SIZE + self.lambd))+(1- self.alpha**2 + self.beta)

        self.weights_mean[:,1:] = 1/ (2*(self.STATE_SIZE + self.lambd))
        self.weights_covariance[1:] = 1/ (2*(self.STATE_SIZE + self.lambd))


    def generateSigmaPoints(self):
        sigma_points = np.zeros((self.n_sigma, self.STATE_SIZE))


        cov_sig = np.sqrt((self.STATE_SIZE+self.lambd)*abs(self.estimateCovariance))

        sigma_points[0] = self.state.T
        for i in range(self.STATE_SIZE):
            sigma_points[i+1] = self.state.T + cov_sig[i]
            sigma_points[i+1+self.STATE_SIZE] = self.state.T - cov_sig[i]

        return sigma_points.T

    def dynamics(self, sigma_points, delta):
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]
        sigma_dynamics = np.empty((self.STATE_SIZE, self.n_sigma))
        sigma_dynamics[0] = self.x+ self.linear_vel* np.cos(sigma_points[2]) * delta
        sigma_dynamics[1] = self.y +self.linear_vel* np.sin(sigma_points[2]) * delta
        sigma_dynamics[2] =  self.theta + self.angular_vel * delta

        return sigma_dynamics

    def computeMean(self, y):
        return np.sum(self.weights_mean*y, axis=1).reshape(-1,1)

    def computeCovariance(self, n1, n2, n3, n4, sigma,size):
        cov_size = size
        covariance = sigma
        for i in range(self.n_sigma):
            diff = (n1[:,i] - n2.T).T
            diff_T = (n3[:,i] - n4.T)
            covariance += self.weights_covariance[i] * np.dot(diff,diff_T)

        return covariance

    def predict(self, delta):
        self.sigmas = self.generateSigmaPoints()

        self.X = self.dynamics(self.sigmas, delta)

        self.state = self.computeMean(self.X)


        self.estimateCovariance = self.computeCovariance(self.X, self.state, self.X, self.state, self.processNoiseCovariance, (3,3))
        self.wrapTheta()
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]

        self.sigmas = self.X

    def update(self):
        update_sig_points = self.generateSigmaPoints()
        self.Z = np.dot(self.observationMatrix, update_sig_points)
        self.z = self.computeMean(self.Z)


        self.innovation_covariance = self.computeCovariance(self.Z, self.z, self.Z, self.z,self.observationCovariance,(2,2))
        self.sigma_t = self.computeCovariance(update_sig_points, self.state, self.Z, self.z, 0, (3,2))

        self.kalman_gain = np.dot(self.sigma_t, inv(self.innovation_covariance))


        self.state = self.state + np.dot(self.kalman_gain, (self.measurement - self.z))
        self.wrapTheta()
        self.estimateCovariance = self.estimateCovariance - multi_dot([self.kalman_gain, self.innovation_covariance, self.kalman_gain.T])
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]

if __name__ == '__main__':

    filter_ = UKF()
    filter_theta_ = UKF()

    estimated_state = np.empty((filter_.true_data.shape[0], 3))

    estimated_state_theta_ = np.empty((filter_.true_data.shape[0], 3))

    filter_theta_.observationMatrix = np.array([[0, 0, 1]]) #H

    filter_theta_.observationCovariance = np.array([[.1**2]], dtype="float64") #R


    t = filter_.true_data

    #infer initial pose from sensor data
    filter_.x = filter_.sensor_data[0][1]
    filter_.y = filter_.sensor_data[0][2]
    filter_.theta = filter_.sensor_data[0][3]
    filter_.setInitState()

    filter_theta_.x = filter_.sensor_data[0][1]
    filter_theta_.y = filter_.sensor_data[0][2]
    filter_theta_.theta = filter_.sensor_data[0][3]
    filter_theta_.setInitState()

    sensor_data = filter_.sensor_data


    for i in range(len(filter_.true_data)):
        current_time = filter_.sensor_data[i][0]
        delta = current_time - filter_.prev_time
        filter_.prev_time = current_time

        #-------------With X and Y measurements only--------------------#
        filter_.predict(delta)
        filter_.measurement = np.array([[filter_.sensor_data[i][1]], [filter_.sensor_data[i][2]]])

        filter_.update()
        filter_.linear_vel = filter_.true_data[i][4]
        filter_.angular_vel = filter_.true_data[i][5]

        estimated_state[i] =filter_.state.T

        #-------------With theta measurements only--------------------#
        filter_theta_.prev_time = current_time
        filter_theta_.predict(delta)
        filter_theta_.measurement = np.array([filter_.sensor_data[i][3]])

        filter_theta_.update()
        filter_theta_.linear_vel = filter_theta_.sensor_data[i][4]
        filter_theta_.angular_vel = filter_theta_.sensor_data[i][5]

        estimated_state_theta_[i] =filter_theta_.state.T


    time = filter_.true_data[:][:,0]
    true_x = filter_.true_data[:][:,1]
    true_y = filter_.true_data[:][:,2]
    true_theta = filter_.true_data[:][:,3]


    est_x = estimated_state[:][:,0]
    est_y = estimated_state[:][:,1]
    est_theta = estimated_state[:][:,2]

    est_xt = estimated_state_theta_[:][:,0]
    est_yt = estimated_state_theta_[:][:,1]
    est_thetat = estimated_state_theta_[:][:,2]

#
#
end_t= len(est_x) -1

#-------------- Root mean square error ----------------#
RMSE_X_1 =  np.sqrt(np.mean((true_x-est_x)**2))
RMSE_Y_1 =  np.sqrt(np.mean((true_y-est_y)**2))
RMSE_Theta_1 =  np.sqrt(np.mean((true_theta-est_theta)**2))

print(RMSE_X_1,RMSE_Y_1, RMSE_Theta_1)


#------------------------- PLOTS ---------------------#

#------------- X AND Y SENSOR MEASUREMENTS ONLY------------#

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Unscented Kalman Filter using X and Y Position Measurements only')

#x and t
ax1.plot(time, true_x, '--r', label="True")
ax1.plot(time, sensor_data[:,1], '--k', label="Measurements", linewidth=0.1)
ax1.plot(time, est_x, '--b', label="Estimated")
ax1.grid(True)
ax1.set(xlabel='Time (secs)', ylabel="X Position (m)")
ax1.set_title('Estimated X Position vs True X Position')
ax1.legend()

#y and t
ax2.plot(time, true_y, '--r', label="True")
ax2.plot(time, sensor_data[:,2], '--k', label="Measurements", linewidth=0.1)
ax2.plot(time, est_y, '--b', label="Estimated")
ax2.grid(True)
ax2.set(xlabel='Time (secs)', ylabel="Y Position (m)")
ax2.set_title('Estimated Y Position vs True Y Position')
ax2.legend()

#theta and t
ax3.plot(time, true_theta, '--r', label="True")
ax3.plot(time, sensor_data[:,3], '--k', label="Measurements", linewidth=0.1)
ax3.plot(time, est_theta, '--b', label="Estimated")
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
ax4.legend()


#------------- THETA SENSOR MEASUREMENT -------------#
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Unscented Kalman Filter using Theta Measurements only')

#x and t
ax1.plot(time, true_x, '--r', label="True")
ax1.plot(time, sensor_data[:,1], '--k', label="Measurements", linewidth=0.1)
ax1.plot(time, est_xt, '--b', label="Estimated")
ax1.grid(True)
ax1.set(xlabel='Time (secs)', ylabel="X Position (m)")
ax1.set_title('Estimated X Position vs True X Position')
ax1.legend()

#y and t
ax2.plot(time, true_y, '--r', label="True")
ax2.plot(time, sensor_data[:,2], '--k', label="Measurements", linewidth=0.1)
ax2.plot(time, est_yt, '--b', label="Estimated")
ax2.grid(True)
ax2.set(xlabel='Time (secs)', ylabel="Y Position (m)")
ax2.set_title('Estimated Y Position vs True Y Position')
ax2.legend()

#theta and t
ax3.plot(time, true_theta, '--r', label="True")
ax3.plot(time, sensor_data[:,3], '--k', label="Measurements", linewidth=0.1)
ax3.plot(time, est_thetat, '--b', label="Estimated")
ax3.grid(True)
ax3.set(xlabel='Time (secs)', ylabel=r"$\bf{\theta}$ (rad)")
ax3.set_title('Estimated Heading vs True Heading')
ax3.legend()

#X and Y
ax4.plot(true_x, true_y, '--r', label="True")
ax4.plot(true_x[0], true_y[0], 'ro', markersize=8.5, label= 'Start')
ax4.plot(true_x[end_t], true_y[end_t], 'bo', markersize=8.5, label ='End')
ax4.plot(est_xt, est_yt, '--b', label="Estimated")
ax4.plot(est_xt[0], est_yt[0], 'ro', markersize=8.5, label= 'Start')
ax4.plot(est_xt[end_t], est_yt[end_t], 'bo', markersize=8.5, label ='End')
ax4.grid(True)
ax4.set(xlabel='X (m)', ylabel="Y (m)")
ax4.set_title('Estimated Robot Trajectory vs True Robot Trajectory')
ax4.legend()
plt.show()
