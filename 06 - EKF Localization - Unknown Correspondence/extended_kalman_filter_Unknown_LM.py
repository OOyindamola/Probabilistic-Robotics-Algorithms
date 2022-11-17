'''
Created on March 27, 2020
@author: Oyindamola Omotuyi
Extended Kalman Filter Localization with Unknown Correspondence Algorithm for a differential drive robot
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
true_data = DATA_DIR+'unknown_correspondence_true_data.csv'
sensor_data = DATA_DIR+'unknown_correspondence_sensor_data.csv'


class EKFLocalizationUnknownCorrespondence:
    def __init__(self):

        self.generate_data()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.state = np.array([[self.x], [self.y], [self.theta]]) #pos, vel, acc
        self.prev_time = 0

        self.estimateCovariance = 0.1**2*np.identity(3) #P


        self.processNoiseCovariance = np.array([[0.1**2,0],
                                                [0, 0.1**2]], dtype="float64") #Q for the control inputs



        self.observationCovariance = np.array([[.4**2, 0, 0],
                                               [0, .2**2, 0],
                                               [0, 0, 1**2]]) #R

        #initial guess
        self.linear_vel = 0.0001
        self.angular_vel = 0.000001
        self.STATE_SIZE = 3



        ##LANDMARKS
        self.landmark_size = 4
        self.landmark_id = 0
        self.map = np.empty((self.landmark_size,3))


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
        self.v_w = self.linear_vel / self.angular_vel

        self.stateTransitionMatrix = np.array([[1,0, self.v_w*(-np.cos(self.theta) + np.cos(self.theta + self.angular_vel*delta))],
                                               [0,1, self.v_w*(-np.sin(self.theta) + np.sin(self.theta + self.angular_vel*delta))],
                                               [0,0,                                        1]
                                              ])


        term_1 = np.sin(self.theta) - np.sin(self.theta + self.angular_vel*delta)
        term_2 = (self.linear_vel/self.angular_vel)*np.cos(self.theta + self.angular_vel*delta)*delta

        term_3 = np.cos(self.theta) - np.cos(self.theta + self.angular_vel*delta)
        term_4 = (self.linear_vel/self.angular_vel)*np.sin(self.theta + self.angular_vel*delta)*delta

        self.inputJacobian = np.array([
                                        [-term_1/self.angular_vel,  (self.linear_vel/self.angular_vel**2)*(term_1) + term_2],
                                        [term_3/self.angular_vel,  (-self.linear_vel/self.angular_vel**2)*(term_3) + term_4 ],
                                        [0,                                                                        delta]
                                        ])

        self.x = self.x - self.v_w*np.sin(self.theta) + self.v_w*np.sin(self.theta + self.angular_vel*delta)
        self.y = self.y + self.v_w*np.cos(self.theta) - self.v_w*np.cos(self.theta + self.angular_vel*delta)
        self.theta = self.theta + self.angular_vel*delta


        self.theta = self.wrapTheta(self.theta)


        self.state = np.array([[self.x],[self.y],[self.theta]])


        self.estimateCovariance = multi_dot([self.stateTransitionMatrix,
                                        self.estimateCovariance,
                                        self.stateTransitionMatrix.T]) + multi_dot([self.inputJacobian,
                                                                                   self.processNoiseCovariance,
                                                                                   self.inputJacobian.T])

        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.theta = self.state[2,0]



    def update(self):
        self.innovation_covariances = np.zeros((self.landmark_size, 3, 3))
        self.mahalobis_distances = np.zeros(self.landmark_size)
        self.observationMatrixes =  np.zeros((self.landmark_size, 3, 3))
        self.innovations = np.zeros((self.landmark_size, 3, 1))

        for i in range(self.landmark_size):
            self.landmark = self.map[i]
            self.landmark_pos = np.array([self.landmark[0], self.landmark[1]])

            self.landmark_signature = self.landmark[2]

            self.predRange = nl.norm(self.landmark_pos-np.array([self.x, self.y]))
            self.predBearing = np.arctan2(self.landmark_pos[1]-self.y, self.landmark_pos[0] - self.x)-self.theta


            self.predMeasurement = np.row_stack((self.predRange, self.predBearing, self.landmark_signature))
            self.observationMatrix = np.array([[-(self.landmark_pos[0] - self.x)/self.predRange, -(self.landmark_pos[1] - self.y)/self.predRange,        0],
                                               [(self.landmark_pos[1] - self.y)/self.predRange**2, -(self.landmark_pos[0] - self.x)/self.predRange**2, -1],
                                               [0,      0,      0]
                                                ])

            self.innovation = self.measurement - self.predMeasurement

            self.innovation_covariance = multi_dot([self.observationMatrix,
                                                self.estimateCovariance,
                                                self.observationMatrix.T]) + self.observationCovariance

            self.innovation_covariances[i] = self.innovation_covariance
            self.observationMatrixes[i] = self.observationMatrix
            self.innovations[i] = self.innovation

            self.mahalobis_distances[i] = det(2*np.pi*self.innovation_covariance)**-0.5 * np.exp(-0.5*
            self.innovation.T@inv(self.innovation_covariance)@self.innovation)


        self.j_i = np.argmax(self.mahalobis_distances)
        self.innovation_covariance = self.innovation_covariances[self.j_i]
        self.observationMatrix = self.observationMatrixes[self.j_i]
        self.innovation = self.innovations[self.j_i]

        self.kalman_gain = multi_dot([self.estimateCovariance,
                                self.observationMatrix.T,
                                inv(self.innovation_covariance)])



        self.state = self.state + np.dot(self.kalman_gain, self.innovation)


        self.identity = np.identity(self.STATE_SIZE)

        first_term = self.identity - np.dot(self.kalman_gain,self.observationMatrix)

        self.estimateCovariance = np.dot(first_term, self.estimateCovariance)
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.theta = self.state[2,0]
        self.theta = self.wrapTheta(self.theta)


if __name__ == '__main__':

    filter_ = EKFLocalizationUnknownCorrespondence()

    estimated_state = np.empty((filter_.true_data.shape[0], 3))

    sensor_data = filter_.sensor_data
    true_data = filter_.true_data

    #Known Landmark Positions but Correspondence is not known
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

            measurements[j][i] = np.array([landmarks_range[j][i], landmarks_bearing[j][i], i])



    for j in range(len(filter_.true_data)):
        current_time = filter_.sensor_data[j][0]
        delta = current_time - filter_.prev_time
        filter_.prev_time = current_time

        filter_.predict(delta)

        for i in range(filter_.landmark_size):

            if(~np.isnan(measurements[j][i][0])):
                filter_.measurement = np.array([measurements[j][i]]).T

                filter_.update()

        filter_.linear_vel = filter_.true_data[j][4]
        filter_.angular_vel = filter_.true_data[j][5]

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
fig.suptitle('EKF Localization (Unknown Correspondence) with Range and Bearing Measurements')

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

ax4.plot(landmarks[0,0],landmarks[0,1], 'k*', markersize=10, label='landmarks')
for i in range(1,len(landmarks)):
    ax4.plot(landmarks[i,0],landmarks[i,1], 'k*', markersize=10)
ax4.legend()
plt.show()
