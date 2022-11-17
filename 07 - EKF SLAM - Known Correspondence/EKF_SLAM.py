'''
Created on Apr 10, 2020
@author: Oyindamola Omotuyi
Extended Kalman Filter Simulateneous Localization and Mapping (SLAM)
    with Known Correspondence Algorithm for a differential drive robot
Robot: Turtlebot3 robot in ROS/Gazebo
'''


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from numpy.linalg import multi_dot
from numpy import linalg as nl
import math

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


class EKFSLAMKnownCorrespondence:
    def __init__(self):

        self.generate_data()

        self.x = 0
        self.y = 0
        self.theta = 0

        self.prev_time = 0

        self.processNoiseCovariance = np.array([[0.01**2,0, 0],
                                                [0, 0.01**2, 0],
                                                [0, 0, 0.01**2]    ], dtype="float64") #Q for the control inputs


        self.observationCovariance = np.array([[.7**2, 0, 0],
                                               [0, .5**2, 0],
                                               [0, 0, 1**2]]) #R

        #initial guess
        self.linear_vel = 0.001
        self.angular_vel = 0.001


        ##LANDMARKS
        self.landmark_size = 4
        self.landmark_id = 0
        self.map = np.empty((self.landmark_size,3))
        self.landmark_positions = np.zeros(3*self.landmark_size)

        self.STATE_SIZE = 3*self.landmark_size + 3
        self.state = np.zeros((self.STATE_SIZE,1)) #pos, vel, acc

        #supply initial position of the robot
        self.state[0,0] = -2
        self.state[1,0] = -0.5
        self.estimateCovariance = 0.1**2*np.identity(self.STATE_SIZE) #P
        self.seenLandmarkIDs = []

    def generate_data(self):
        self.true_data= np.genfromtxt(true_data, delimiter=',')
        self.sensor_data= np.genfromtxt(sensor_data, delimiter=',')


        self.true_data = np.array(self.true_data[1:], dtype=np.float64)
        self.sensor_data = np.array(self.sensor_data[1:], dtype=np.float64)

    def wrapTheta(self, angle):
        if(angle >  2*np.pi):
            print("angle", angle)
            angle -= 2*np.pi
            print("final", angle)
        return angle

    def predict(self, delta):
        self.Fx = np.column_stack((np.identity(3), np.zeros((3,3*self.landmark_size))))


        self.v_w = self.linear_vel / self.angular_vel

        self.theta = self.state[2,0]

        self.stateDynamics = np.array([[- self.v_w*np.sin(self.theta) + self.v_w*np.sin(self.theta + self.angular_vel*delta)],
                                       [self.v_w*np.cos(self.theta) - self.v_w*np.cos(self.theta + self.angular_vel*delta)],
                                        [self.angular_vel*delta]])

        self.state = self.state + self.Fx.T@self.stateDynamics


        self.stateTransitionMatrix = np.array([[0,0, self.v_w*(-np.cos(self.theta) + np.cos(self.theta + self.angular_vel*delta))],
                                               [0,0, self.v_w*(-np.sin(self.theta) + np.sin(self.theta + self.angular_vel*delta))],
                                               [0,0,                                        0]
                                              ])


        self.Gt = np.identity(self.STATE_SIZE) + self.Fx.T@self.stateTransitionMatrix@self.Fx

        self.estimateCovariance = multi_dot([self.Gt,
                                        self.estimateCovariance,
                                        self.Gt.T]) + multi_dot([self.Fx.T,
                                                                 self.processNoiseCovariance,
                                                                   self.Fx])

    def update(self):

        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.theta = self.state[2,0]

        r = self.measurement[0,0]
        phi = self.measurement[1,0]
        signature = self.measurement[2,0]

        j = int(signature)

        if not j in self.seenLandmarkIDs:
            self.state[3*j : 3*j + 3]  = np.array([[self.x], [self.y],
                                                            [signature]]) + np.array([[ r* np.cos(phi + self.theta)],
                                                                                    [r* np.sin(phi + self.theta)],
                                                                                    [0]])
            self.seenLandmarkIDs.append(j)


        self.estimatedLandmark = self.state[3*j : 3*j + 3]

        self.landmark_pos =  np.array([self.estimatedLandmark[0,0], self.estimatedLandmark[1,0]])




        self.delta_x = self.landmark_pos[0] - self.x
        self.delta_y = self.landmark_pos[1] - self.y

        self.delta = np.array([[self.delta_x], [self.delta_y]])

        q = self.delta.T @ self.delta

        self.pRange = np.sqrt(q)

        robot_pos = np.array([self.x, self.y])

        self.predRange = nl.norm(self.landmark_pos-robot_pos)
        self.predBearing = np.arctan2(self.delta_y, self.delta_x)-self.theta

        self.predMeasurement = np.row_stack((self.predRange, self.predBearing, self.state[3*j + 2] ))

        self.Fxj = np.zeros((6, self.STATE_SIZE))

        self.Fxj[0,0] = 1
        self.Fxj[1,1] = 1
        self.Fxj[2,2] = 1

        self.Fxj[3,3*j] = 1
        self.Fxj[4,3*j+1] = 1
        self.Fxj[5,3*j+2] = 1


        self.observationMatrix = (1/(self.predRange**2)) * np.array([[-self.predRange*self.delta_x, -self.predRange*self.delta_y, 0, self.predRange*self.delta_x, self.predRange*self.delta_y, 0],
                                                                     [self.delta_y,               -self.delta_x,   -self.predRange**2, -self.delta_y,               self.delta_x,                0],
                                                                     [0,0,0,0,0, self.predRange**2]
                                                                    ]) @ self.Fxj

        self.innovation = self.measurement - self.predMeasurement

        self.innovation_covariance = multi_dot([self.observationMatrix,
                                            self.estimateCovariance,
                                            self.observationMatrix.T]) + self.observationCovariance

        self.kalman_gain = multi_dot([self.estimateCovariance,
                                self.observationMatrix.T,
                                inv(self.innovation_covariance)])


        self.state = self.state + np.dot(self.kalman_gain, self.innovation)

        self.identity = np.identity(self.STATE_SIZE)

        first_term = self.identity - np.dot(self.kalman_gain,self.observationMatrix)

        self.estimateCovariance = np.dot(first_term, self.estimateCovariance)


if __name__ == '__main__':

    filter_ = EKFSLAMKnownCorrespondence()

    estimated_state = np.empty((filter_.true_data.shape[0], 15))

    sensor_data = filter_.sensor_data
    true_data = filter_.true_data

    #Known Landmark Positions but this is not known to the robot
    landmarks = np.array([[-2.5,2.5], [-2.5,-1.0], [2.5,-1.0], [2.5, 2.5]])
    filter_.landmark_size = landmarks.shape[0]

    filter_.map = np.zeros((filter_.landmark_size,3))
    measurements = np.zeros((filter_.true_data.shape[0], filter_.landmark_size, 3))

    landmarks_range = np.column_stack((filter_.sensor_data[:,1], filter_.sensor_data[:,3], filter_.sensor_data[:,5], filter_.sensor_data[:,7]))

    landmarks_bearing = np.column_stack((filter_.sensor_data[:,2], filter_.sensor_data[:,4], filter_.sensor_data[:,6], filter_.sensor_data[:,8]))

    ##DATA GENERATION
    for j in range(len(filter_.true_data)):
        for i in range(len(landmarks)):
            filter_.map[i] = np.array([landmarks[i,0], landmarks[i,1], i])

            measurements[j][i] = np.array([landmarks_range[j][i], landmarks_bearing[j][i], i+1])



    for j in range(len(filter_.true_data)):
        current_time = filter_.sensor_data[j][0]
        delta = current_time - filter_.prev_time
        filter_.prev_time = current_time

        filter_.predict(delta)

        for i in range(filter_.landmark_size):
            if(~np.isnan(measurements[j][i][0])):
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


end_t= len(est_x) -1

#------------------------- PLOTS ---------------------#

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('EKF SLAM (Known Correspondence) with Range and Bearing Measurements')

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
