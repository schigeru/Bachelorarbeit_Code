#!/usr/bin/env python3
'''Gym_Environment zur Steuerung des Panda in Gazebo durch RL-Agent'''
import math
import numpy as np
import os
import time
import sys
import copy
import rospy
import moveit_msgs.msg
import geometry_msgs.msg
import random
import csv
import gym
from gym import spaces
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.msg import LinkState
from std_msgs.msg import Float64
from std_msgs.msg import String
from panda_rl.srv import ResetJoints, ResetJointsResponse
from panda_rl.srv import StepAction, StepActionResponse
from geometry_msgs.msg import PointStamped
metadata = {'render.modes': ['human']}

pub = rospy.Publisher('test_point', PointStamped, queue_size=10)
rospy.init_node('gym_env')


class PandaRobotGymEnv(gym.Env):

    def __init__(self, max_steps=50):
        super(PandaRobotGymEnv, self).__init__()
        self.stepnode = rospy.ServiceProxy('step_env', StepAction, persistent=True)
        self.res = rospy.ServiceProxy('reset_env', ResetJoints, persistent=True)
        self._env_step_counter = 0
        self.done = False
        self._max_steps = max_steps
        self.observation_space = spaces.Box(np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -2, -2, -2]),
                                            np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 2.8973, 3.7525, 2, 2, 2]))
        self.action_space = spaces.Box(np.array([-1, -1, -1, -1, -1, -1]), np.array([+1, +1, +1, +1, +1, +1]))
        self.goal = np.array([0, 0, 0])
        self.goal_point = PointStamped()

    def reset(self):
        rospy.wait_for_service("reset_env")
        try:
            goalx = random.randrange(8, 13) / 20 #Diskretisierte zufällige Zielpunkte - Versch. Werte getestet
            goaly = random.randrange(-4, 5) / 20
            goalz = random.randrange(6, 11) / 20
            self.goal_point.header.seq = 1 #Visualisierung des Zielpunkts in Rviz
            self.goal_point.header.stamp = rospy.Time.now() #Visualisierung des Zielpunkts in Rviz
            self.goal_point.header.frame_id = "world" #Visualisierung des Zielpunkts in Rviz
            self.goal_point.point.x = goalx #Visualisierung des Zielpunkts in Rviz
            self.goal_point.point.y = goaly #Visualisierung des Zielpunkts in Rviz
            self.goal_point.point.z = goalz #Visualisierung des Zielpunkts in Rviz

            pub.publish(self.goal_point)

            self.goal = [goalx, goaly, goalz]

            response = self.res(self.goal)
            self._env_step_counter = 0
            self.done = False
            return np.array(response.obs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def step(self, action):
        rospy.wait_for_service("step_env")
        try:
            response = self.stepnode(action, self.goal)
            obs = response.obs
            reward = response.reward
            self.done = response.done
            self._env_step_counter += 1

            if self._env_step_counter >= self._max_steps:
                reward = 0
                self.done = True

            return np.array(obs), np.array(reward), np.array(self.done), {}
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def render(self, mode='human'):
        print(self.done, self._env_step_counter)
