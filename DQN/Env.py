import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Tools.AirsimTools as airsimtools
import airsim

DOWN_SENSOR_IDX = 6
INITIAL_DISTANCE = -1
ROUND_DECIMALS = 2
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

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    return [client.getDistanceSensorData(DISTANCE_SENSOR["f"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["l"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["r"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["lf"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["rf"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["t"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["b"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["lfb"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["rfb"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["lbb"], drone_name).distance,
                    client.getDistanceSensorData(DISTANCE_SENSOR["rbb"], drone_name).distance]


class AirsimDroneEnv(gym.Env):
    def __init__(self, reward_function, sensor_num, client:airsim.MultirotorClient):
        super(AirsimDroneEnv, self).__init__()
        self.sensor_num = sensor_num
        self.client = client
        self.action_space = spaces.Box(low=np.array([-100, -100, -100]), high=np.array([100, 100, 100]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(self.sensor_num), high=np.array([20]*self.sensor_num), dtype=np.float32)
        self.state = np.full(self.sensor_num, 20)
        self.reward_function = reward_function
        self.prev_dis = INITIAL_DISTANCE

    def reset(self):
        self.state = np.full(self.sensor_num, 20)
        self.prev_dis = INITIAL_DISTANCE

        return self.state, {}
    
    def step(self, action, targets, step_cnt, drone_name):
        '''
        senor_values: [front, left, right, front_left, front_right, top, bottom]
        targets:[[n, e, d]]
        drone_position:[n, e, d]
        '''
        n, e, d = action
        n, e, d = airsimtools.scale_and_normalize_vector([n, e, d], 1)
        if step_cnt <= 0: # before action, take off the drone
            self.client.takeoffAsync(5, drone_name).join()
        self.client.moveByVelocityAsync(float(n), float(e), float(d), 0.1).join() # drone move

        sensor_data = get_distance_sensor_data(self.client, drone_name = drone_name)
        drone_position = self.client.simGetVehiclePose(drone_name).position
        velocity = airsimtools.scale_and_normalize_vector(airsimtools.get_velocity(drone_position, targets[0], 2), 1)
        drone_position = airsimtools.check_negative_zero(np.round(drone_position.x_val, ROUND_DECIMALS), np.round(drone_position.y_val, ROUND_DECIMALS), np.round(drone_position.z_val, ROUND_DECIMALS))
        drone_data = velocity + sensor_data
        # get new state
        self.state = np.array(drone_data)
        done, overlap = self.check_done(drone_data, targets, drone_position, step_cnt)
        curr_dis = airsimtools.calculate_distance(drone_position, targets[0])
        if curr_dis < 0.5:
            del targets[0]
        if len(targets) > 0:
            reward, info = self.reward_function(done, overlap, self.prev_dis, curr_dis)
        else:
            reward, info = self.reward_function(done, overlap, self.prev_dis, curr_dis)

        self.prev_dis = info['prev_dis']

        return self.state, reward, done, False, {'velocity': action, 'overlap': overlap, 'targets' : targets}

    def check_done(self, sensor_values, targets, drone_position, step_cnt):
        # Check if all values ​​except down_distance are less than 0.1
        overlap = any(sensor < 0.05 for i, sensor in enumerate(sensor_values) if i != DOWN_SENSOR_IDX)
        distance = airsimtools.calculate_distance(drone_position, targets[0])
        if overlap or distance > 100 or step_cnt > 500:
            return True, True
        elif len(targets) == 0:
            return True, False
        else:
            return False, False

    def render(self, mode='human'):
        print(f'State: {self.state}')