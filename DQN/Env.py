import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Tools.AirsimTools as airsimtools

DOWN_SENSOR_IDX = 6
INITIAL_DISTANCE = -1

class AirsimDroneEnv(gym.Env):
    def __init__(self, reward_function, sensor_num):
        super(AirsimDroneEnv, self).__init__()
        self.sensor_num = sensor_num
        self.action_space = spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(self.sensor_num), high=np.array([20]*self.sensor_num), dtype=np.float32)
        self.state = np.full(self.sensor_num, 20)
        self.reward_function = reward_function
        self.prev_dis = INITIAL_DISTANCE

    def reset(self):
        self.state = np.full(self.sensor_num, 20)
        self.prev_dis = INITIAL_DISTANCE

        return self.state, {}
    
    def step(self, action, sensor_values, targets, drone_position, step_cnt):
        '''
        senor_values: [front, left, right, front_left, front_right, top, bottom]
        targets:[[n, e, d]]
        drone_position:[n, e, d]
        '''
        # get new state
        self.state = np.array(sensor_values)
        done, overlap = self.check_done(sensor_values, targets, drone_position, step_cnt)
        if drone_position == targets[0]:
            del targets[0]
        if len(targets) > 0:
            reward, info = self.reward_function(self.state, action, self.state, done, overlap, self.prev_dis, drone_position, targets[0])
        else:
            reward, info = self.reward_function(self.state, action, self.state, done, overlap, self.prev_dis, drone_position, drone_position)

        self.prev_dis = info['prev_dis']

        return self.state, reward, done, False, {'velocity': action, 'overlap': overlap, 'targets' : targets}

    def check_done(self, sensor_values, targets, drone_position, step_cnt):
        # Check if all values ​​except down_distance are less than 0.1
        overlap = any(sensor < 0.1 for i, sensor in enumerate(sensor_values) if i != DOWN_SENSOR_IDX)
        distance = airsimtools.calculate_distance(drone_position, targets[0])
        if overlap or distance > 200 or step_cnt > 500:
            return True, True
        elif len(targets) == 0:
            return True, False
        else:
            return False, False

    def render(self, mode='human'):
        print(f'State: {self.state}')