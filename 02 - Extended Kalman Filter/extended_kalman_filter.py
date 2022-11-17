'''
Created on Feb 7, 2020
@author: Oyindamola Omotuyi
Extended Kalman Filter Algorithm for a differential drive robot
Robot: Turtlebot3 robot in ROS/Gazebo
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import multi_dot


import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'


#Set Data file variables
DATA_DIR = "./data/"
true_data = DATA_DIR+'true_data.csv'
sensor_data = DATA_DIR+'sensor_data.csv'


class EKF:
    def __init__(self):

        self.generate_data()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.state = np.array([[self.x, self.y, self.theta]]) #pos, vel, acc
        self.prev_time = 0

        self.estimateCovariance = np.identity(3) #P

        self.processNoiseCovariance = np.array([[.1**2, 0],
                                                [0, .1**2]]) #Q for the control inputs


        self.observationMatrix = np.array([[1, 0, 0],
                                           [0, 1, 0]]) #H

        self.observationCovariance = np.array([[.5**2, 0],
                                               [0, .5**2]]) #R


        self.linear_vel = 0
        self.angular_vel = 0
        self.STATE_SIZE = 3
        self.count = 0


    def generate_data(self):
        self.true_data= np.genfromtxt(true_data, delimiter=',')
        self.sensor_data= np.genfromtxt(sensor_data, delimiter=',')

        self.true_data = np.array(self.true_data[1:], dtype=np.float64)
        self.sensor_data = np.array(self.sensor_data[1:], dtype=np.float64)

    def setInitState(self):
        self.state = np.array([[self.x], [self.y], [self.theta]])

    def setInitState_(self, x, y, theta):
        self.state = np.array([[x], [y], [theta]])

    def wrapTheta(self, angle):
        if(angle >  2*np.pi):
            print("angle", angle)
            angle -= 2*np.pi
            print("final", angle)
        return angle

    def predict(self, delta):

        self.stateTransitionMatrix = np.array([[1,0, -delta*self.linear_vel*np.sin(self.theta) ],
                                               [0,1, delta*self.linear_vel*np.cos(self.theta) ],
                                               [0,0,                                        1]
                                              ])

        self.inputJacobian = np.array([
                                        [delta*np.cos(self.theta), 0],
                                        [delta*np.sin(self.theta), 0],
                                        [0, delta]
                                        ])

        self.x = self.x + self.linear_vel* np.cos(self.theta) * delta
        self.y = self.y + self.linear_vel * np.sin(self.theta) *delta
        self.theta = self.theta + self.angular_vel*delta


        self.theta = self.wrapTheta(self.theta)

        self.state = np.array([self.x,self.y,self.theta])

        self.estimateCovariance = multi_dot([self.stateTransitionMatrix,
                                        self.estimateCovariance,
                                        np.transpose(self.stateTransitionMatrix)]) + multi_dot([self.inputJacobian,
                                                                                                self.processNoiseCovariance,
                                                                                                np.transpose(self.inputJacobian)])

        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]



    def update(self):

        self.innovation = self.measurement - np.dot(self.observationMatrix,  self.state) # y = z - Hx

        self.innovation_covariance = multi_dot([self.observationMatrix,
                                            self.estimateCovariance,
                                            np.transpose(self.observationMatrix)]) + self.observationCovariance

        self.kalman_gain = multi_dot([self.estimateCovariance,
                                np.transpose(self.observationMatrix),
                                inv(self.innovation_covariance)])

        self.state = self.state + np.dot(self.kalman_gain, self.innovation)

        self.identity = np.identity(self.STATE_SIZE)

        first_term = self.identity - np.dot(self.kalman_gain,self.observationMatrix)

        self.estimateCovariance = np.dot(first_term, self.estimateCovariance)
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[2]

if __name__ == '__main__':

    filter_ = EKF()
    filter_theta_ = EKF()

    estimated_state = np.empty((filter_.true_data.shape[0], 3))

    estimated_state_theta_ = np.empty((filter_.true_data.shape[0], 3))

    filter_theta_.observationMatrix = np.array([[0, 0, 1]]) #H

    filter_theta_.observationCovariance = np.array([[2**2]]) #R


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
        filter_.measurement = np.array([filter_.sensor_data[i][1], filter_.sensor_data[i][2]])

        filter_.update()
        filter_.linear_vel = filter_.true_data[i][4]
        filter_.angular_vel = filter_.true_data[i][5]

        estimated_state[i] =filter_.state

        #-------------With theta measurements only--------------------#
        filter_theta_.prev_time = current_time
        filter_theta_.predict(delta)
        filter_theta_.measurement = np.array([filter_.sensor_data[i][3]])

        filter_theta_.update()
        filter_theta_.linear_vel = filter_theta_.sensor_data[i][4]
        filter_theta_.angular_vel = filter_theta_.sensor_data[i][5]

        estimated_state_theta_[i] =filter_theta_.state


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
#------------------------- PLOTS ---------------------#

#------------- X AND Y SENSOR MEASUREMENTS ONLY------------#
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Extended Kalman Filter using X and Y Position Measurements only')

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
fig.suptitle('Extended Kalman Filter using Theta Measurements only')

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
