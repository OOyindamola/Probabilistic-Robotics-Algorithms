'''
Created on Jan 31, 2020
@author: Oyindamola Omotuyi
Kalman Filter Algorithm for a one dimensional target tracking problem
'''

import numpy as np
import matplotlib.pyplot as plt
import csv
from numpy.linalg import inv
from numpy.linalg import multi_dot
from os import path
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['figure.titlesize'] = 15
mpl.rcParams['legend.fontsize'] = 15
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'

np.random.seed(1)

#Set Data file variables
DATA_DIR = "./data/"
true_data = DATA_DIR+'true_data.csv'
sensor_pos_data = DATA_DIR+'sen_pos_data.csv'
sensor_acc_data = DATA_DIR+'sen_acc_data.csv'


#set noise parameters for sensor data
noise_sigma = 4
noise_sigma_acc = .5

class EKF:
    def __init__(self):

        self.generate_data()
        self.pos = 0
        self.vel = 0
        self.acc = 0
        self.state = np.array([self.pos, self.vel, self.acc]) #pos, vel, acc

        self.estimateCovariance = np.identity(3) #P


        self.observationMatrix = np.array([[1, 0, 0]]) #H

        self.observationCovariance = np.array([[1]]) #R

        self.STATE_SIZE = 3


    def generate_data(self):
        self.true_data = []
        with open(true_data) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                self.true_data = row
                line_count += 1
        #true_data = np.transpose(true_data)


        self.sen_pos_data = []
        with open(sensor_pos_data) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                self.sen_pos_data =row
                line_count += 1

        self.sen_acc_data = []
        with open(sensor_acc_data) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                self.sen_acc_data=row
                line_count += 1

        self.true_data = np.array(self.true_data, dtype=np.float32)
        self.sen_pos_data = np.array(self.sen_pos_data, dtype=np.float32) + np.random.normal(0, noise_sigma, len(self.sen_pos_data))
        self.sen_acc_data = np.array(self.sen_acc_data, dtype=np.float32)+ np.random.normal(0,noise_sigma_acc, len(self.sen_acc_data))

    def predict(self,delta):
        F = np.array([[1,    delta,  0.5*(delta**2)],
                      [0,    1,      delta],
                      [0,    0,       1]])



        self.processNoiseCovariance = 0.1**2 *np.array([[(delta**4)/4,    (delta**3)/2,      (delta**2)/2],
                                                [(delta**3)/2,    2*(delta**3),       delta**2],
                                                [(delta**2)/2,    delta**2,           delta**2]]) #Q

        self.state =  np.dot(F,self.state)


        self.estimateCovariance = multi_dot([F, self.estimateCovariance, np.transpose(F)]) + self.processNoiseCovariance



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

        self.estimateCovariance = multi_dot([first_term,
                                            self.estimateCovariance,
                                            np.transpose(first_term)]) + multi_dot([self.kalman_gain,
                                                                                    self.observationCovariance,
                                                                                    np.transpose(self.kalman_gain)])


if __name__ == '__main__':

    # dt
    delta = 0.1

    #simulation duration
    T = 20

    estimated_state_pos = np.ones((200,3))
    estimated_state_acc = np.ones((200,3))
    time_steps = np.linspace(0,T,200)
    filter_with_pos_ = EKF()
    filter_with_acc_ = EKF()


    filter_with_pos_.observationMatrix = np.array([[1, 0, 0]]) #H
    filter_with_pos_.estimateCovariance = 2*np.identity(3) #P
    filter_with_pos_.estimateCovariance[1][1]= 1
    filter_with_pos_.estimateCovariance[2][2]= 1

    filter_with_acc_.observationMatrix = np.array([[0, 0, 1]]) #H

    filter_with_acc_.estimateCovariance = 1*np.identity(3) #P
    filter_with_acc_.estimateCovariance[1][1]= 1
    filter_with_acc_.estimateCovariance[2][2]=1

    prev_time = 0


    for i in range(len(filter_with_pos_.true_data)):
        current_time = time_steps[i]
        delta = current_time - prev_time

        filter_with_pos_.predict(delta)
        filter_with_acc_.predict(delta)

        prev_time = current_time

        filter_with_pos_.measurement = np.array([filter_with_pos_.sen_pos_data[i] ])
        filter_with_acc_.measurement = np.array([filter_with_acc_.sen_acc_data[i]])


        filter_with_pos_.update()
        filter_with_acc_.update()

        estimated_state_pos[i] = filter_with_pos_.state
        estimated_state_acc[i] = filter_with_acc_.state


    est_pos_pos = estimated_state_pos[:,0]
    est_vel_pos = estimated_state_pos[:,1]
    est_acc_pos = estimated_state_pos[:,2]

    est_pos_acc = estimated_state_acc[:,0]
    est_vel_acc = estimated_state_acc[:,1]
    est_acc_acc = estimated_state_acc[:,2]


end_t = len(time_steps) -1
true_data = filter_with_pos_.true_data


#------------------- PLOTS ---------------------#
fig, ((ax1, ax2)) = plt.subplots(1, 2)
fig.suptitle('Kalman Filter (1-D target)')

#Position Data with Position Estimated Measurements
ax1.plot(time_steps, true_data, '--r', label="True")
ax1.plot(time_steps, filter_with_pos_.sen_pos_data, '--k', label="Measurements", linewidth=0.9)
ax1.plot(time_steps, est_pos_pos, '--b', label="Estimated", linewidth=2.5)
ax1.grid(True)
ax1.set(xlabel='Time (secs)', ylabel="Position (m)")
ax1.set_title('Estimated Position vs True Position using Position Measurements')
ax1.legend()

#Position Data with Acceleration Estimated Measurements
ax2.plot(time_steps, true_data, '--r', label="True")
ax2.plot(time_steps, filter_with_pos_.sen_pos_data, '--k', label="Measurements", linewidth=0.9)
ax2.plot(time_steps, est_pos_acc, '--b', label="Estimated", linewidth=2.5)
ax2.grid(True)
ax2.set(xlabel='Time (secs)', ylabel="Position (m)")
ax2.set_title('Estimated Position vs True Position using Acceleration Measurements')
ax2.legend()


plt.show()
