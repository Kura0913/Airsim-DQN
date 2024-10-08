import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Tools.AirsimTools as airsimtools
import airsim


DOWN_SENSOR_IDX = 6
INITIAL_DISTANCE = -1
ROUND_DECIMALS = 2
DISTANCE_RANGE = (0, 5)
MAPING_RANGE = (1, 3)
DISTANCE_SENSOR = {
    "f" : "front",
    "l" : "left",
    "r" : "right",
    "rf" : "rfront",
    "lf" : "lfront",
    "t" : "top",
    "b" : "bottom",
    'lfb': 'lfbottom',
    'rfb': 'rfbottom',
    'lbb': 'lbbottom',
    'rbb': 'rbbottom'    
}

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name, sensor_list):
    sensor_data = []
    for sensor_name in sensor_list:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data

class AirsimDroneEnv(gym.Env):
    def __init__(self, reward_function, state_dim, client:airsim.MultirotorClient, drone_name, sensor_list):
        super(AirsimDroneEnv, self).__init__()
        self.state_dim = state_dim
        self.client = client
        self.drone_name = drone_name
        self.action_space = spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(self.state_dim), high=np.array([20]*self.state_dim), dtype=np.float32)
        self.state = np.full(self.state_dim, 20)
        self.reward_function = reward_function
        self.prev_dis = INITIAL_DISTANCE
        self.sensor_list = sensor_list

    def reset(self, curr_target):
        drone_position = self.client.simGetVehiclePose(self.drone_name).position
        drone_position = airsimtools.check_negative_zero(np.round(drone_position.x_val, ROUND_DECIMALS), np.round(drone_position.y_val, ROUND_DECIMALS), np.round(drone_position.z_val, ROUND_DECIMALS))
        self.state = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                drone_position[0], drone_position[1], drone_position[2], 
                                curr_target[0], curr_target[1], curr_target[2],
                               ], dtype=np.float32)
        self.prev_dis = INITIAL_DISTANCE
        
        return self.state, {}
    
    def step(self, action, targets, step_cnt, drone_name):
        '''
        targets:[[n, e, d]]
        '''
        n, e, d = action
        speed = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, self.prev_dis)
        n, e, d = airsimtools.scale_and_normalize_vector([n, e, d], speed)
        if step_cnt <= 0: # before action, take off the drone
            self.client.takeoffAsync(5, drone_name).join()
        self.client.moveByVelocityAsync(float(n), float(e), float(d), 0.15, yaw_mode=airsimtools.get_yaw_mode_F([float(n), float(e), float(d)])).join() # drone move

        sensor_data = get_distance_sensor_data(self.client, drone_name = drone_name, sensor_list=self.sensor_list)
        drone_position = self.client.simGetVehiclePose(drone_name).position
        drone_position = airsimtools.check_negative_zero(np.round(drone_position.x_val, ROUND_DECIMALS), np.round(drone_position.y_val, ROUND_DECIMALS), np.round(drone_position.z_val, ROUND_DECIMALS))
        drone_data = sensor_data + drone_position + targets[0]
        # get new state
        self.state = np.array(drone_data)        
        curr_dis = airsimtools.calculate_distance(drone_position, targets[0])
        if curr_dis < 0.5:
            arrive_target = True
            del targets[0]
        else:
            arrive_target = False
        
        done, overlap = self.check_done(drone_data, targets, drone_position, step_cnt)

        reward, info = self.reward_function(done, overlap, self.prev_dis, curr_dis, arrive_target)

        self.prev_dis = info['prev_dis']

        return self.state, reward, done, False, {'velocity': action, 'overlap': overlap, 'targets' : targets}

    def check_done(self, sensor_values, targets, drone_position, step_cnt):
        if len(targets) == 0:
            return True, False
        # Check if all values ​​except down_distance are less than 0.1
        overlap = any(sensor < 0.05 and sensor != -1 for i, sensor in enumerate(sensor_values) if i not in {DOWN_SENSOR_IDX, 11, 12, 13, 14, 15, 16})
        distance = airsimtools.calculate_distance(drone_position, targets[0])
        if overlap or distance > 100 or step_cnt > 1000:
            return True, True            
        else:
            return False, False

    def render(self, mode='human'):
        print(f'State: {self.state}')