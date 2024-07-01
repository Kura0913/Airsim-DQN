from DQN.DQNAgent import DQNAgent
from DQN.Env import AirsimDroneEnv
from ShortestPath import TravelerShortestPath as tsp
import Tools.AirsimTools as airsimtools
import Tools.DQNTools as dqntools
import numpy as np
import airsim
import os
import json
import sys
import argparse
import time

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
    'rbb': 'rbbottom',
    
}

OBJECT_NAME = "BP_Grid"
ROUND_DECIMALS = 2
BASE_PTAH = '.\\execute\\runs\\'
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

def calculate_reward(state, action, next_state, done, overlap, prev_dis, drone_position, curr_target):
    if done:
        if overlap:
            return -100, {'prev_dis' : -1}
        else:
            return 100, {'prev_dis' : -1}
    else:
        curr_distance = airsimtools.calculate_distance(drone_position, curr_target)
        if prev_dis < 0 or curr_distance < prev_dis:
            return 1, {'prev_dis' : curr_distance}
        else:
            return -2, {'prev_dis' : curr_distance}

def get_targets(client:airsim.MultirotorClient, targets, round_decimals):

    target_pos_ary = []

    for target in targets:
        target_pos = client.simGetObjectPose(target).position
        target_pos = [np.round(target_pos.x_val, round_decimals), np.round(target_pos.y_val, round_decimals), np.round(target_pos.z_val, round_decimals)]
        target_pos = airsimtools.check_negative_zero(target_pos[0], target_pos[1], target_pos[2])
        target_pos_ary.append(target_pos)
    
    drone_pos = client.simGetVehiclePose(drone_name).position
    drone_pos = airsimtools.check_negative_zero(np.round(drone_pos.x_val, round_decimals), np.round(drone_pos.y_val, round_decimals), np.round(drone_pos.z_val, round_decimals))
    target_pos_ary = tsp.getTSP(target_pos_ary, drone_pos)
    del target_pos_ary[0]
    print('best path:', target_pos_ary)
    return target_pos_ary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distance between two coordinates.")
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--episodes', type=int, default=1000, help='number of training')
    parser.add_argument('--gamma', type=float, default=0.99, help='weight of previous reward')
    parser.add_argument('--weight', type=str, default='', help='weight path')
    args = parser.parse_args()
    
    user_home = os.path.expanduser('~')
    settings_path = os.path.join(user_home, 'Documents', 'AirSim', 'settings.json')
    with open(settings_path, 'r') as file:
        data = json.load(file)
    vehicle_names = []
    vehicles = data.get('Vehicles', {})
    for vehicle, _ in vehicles.items():
        vehicle_names.append(vehicle)

    if len(vehicle_names) > 0:
        drone_name = vehicle_names[0]
        client = airsim.MultirotorClient()
        client.confirmConnection()        
        sensor_num = len(get_distance_sensor_data(client, drone_name))
        env = AirsimDroneEnv(calculate_reward, sensor_num)
        agent = DQNAgent(state_dim=sensor_num, action_dim=3, bacth_size=args.batch_size, gamma=args.gamma)
        episodes = args.episodes

        objects = client.simListSceneObjects(f'{OBJECT_NAME}[\w]*')
        targets = get_targets(client, objects, ROUND_DECIMALS)
        
        if len(targets) > 0:
            if args.weight != '':
                try:
                    agent.load(args.weight)
                except:
                    print(f"The path:{args.weight} is not exist, load weight fail.")
            
            for episode in range(episodes):
                client.reset()
                client.enableApiControl(True)
                state, _ = env.reset()
                done = False
                rewards = 0
                step_count = 0
                while not done:
                    action = agent.act(state)
                    n, e, d = action
                    n, e, d = airsimtools.scale_and_normalize_vector([n, e, d], 1)
                    if step_count <= 0:
                        client.takeoffAsync(10, drone_name).join()

                    client.moveByVelocityAsync(float(n), float(e), float(d) , 0.1).join()
                    
                    sensor_values = get_distance_sensor_data(client, drone_name)
                    drone_pos = client.simGetVehiclePose().position
                    drone_pos = airsimtools.check_negative_zero(np.round(drone_pos.x_val, ROUND_DECIMALS), np.round(drone_pos.y_val, ROUND_DECIMALS), np.round(drone_pos.z_val, ROUND_DECIMALS))
                    next_state, reward, done, _, info = env.step(action, sensor_values, targets = targets, drone_position=drone_pos)                
                    agent.store_experience(state, action, reward, next_state, done)
                    state = next_state
                    
                    agent.train()
                    rewards += reward # calculate total rewards
                    step_count += 1
                    if done:
                        if info['overlap']:
                            status = (f'Episode: {episode:3d}/{episodes} | Step: {step_count} | Reward: {rewards:3d} | mission_state: fail')
                        else:
                            status = (f'Episode: {episode:3d}/{episodes} | Step: {step_count} | Reward: {rewards:3d} | mission_state: success')
                    else:
                        status = (f'Episode: {episode:3d}/{episodes} | Step: {step_count} | Reward: {rewards:3d} | mission_state: run')
                    sys.stdout.write('\r' + status)
                    sys.stdout.flush()



                print(f'\r')

            agent.save(f"{dqntools.create_directory(BASE_PTAH)}\\model.pth") # save weight
            print("Updated model saved!")
        else:
            print("The corresponding object cannot be found in the environment and training cannot be started.")
