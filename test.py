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
import threading
import signal
import time

DISTANCE_SENSOR = ["front", "left", "right", "rfront", "lfront", "top", "bottom", 'lfbottom', 'rfbottom', 'lbbottom', 'rbbottom']

ROUND_DECIMALS = 2
DRONE_BOTTOM_LIMIT = 2
DRONE_POSITION_LEN = 3
TARGET_POSITION_LEN = 3
BASE_PTAH = '.\\execute\\runs\\'

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    sensor_data = []
    for sensor_name in DISTANCE_SENSOR:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data

def check_file_exists(file_path):
    if os.path.exists(file_path):
        print(f"The file {file_path} exists.")

        return True
    else:
        print(f"The file {file_path} does not exist.")

        return False

def signal_handler(signum, frame):
    global stop_event
    print("\nTraining interrupted...")
    print("Exiting...")
    stop_event.set()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSim-DQN test.")
    parser.add_argument('--episodes', type=int, default=5, help='number of training')
    parser.add_argument('--weight', type=str, default='', help='weight path')
    parser.add_argument('--object', type=str, default='BP_Grid', help='The object name in the vr environment, you can place objects in the VR environment and make sure that the objects you want to visit start with the same name.. Initial object is: BP_Grid')
    args = parser.parse_args()
    # to stop training and save the weight    
    stop_event = threading.Event()

    user_home = os.path.expanduser('~')
    settings_path = os.path.join(user_home, 'Documents', 'AirSim', 'settings.json')
    with open(settings_path, 'r') as file:
        data = json.load(file)
    vehicle_names = []
    vehicles = data.get('Vehicles', {})
    for vehicle, _ in vehicles.items():
        vehicle_names.append(vehicle)

    if len(vehicle_names) > 0 or args.weight == '':
        drone_name = vehicle_names[0]
        client = airsim.MultirotorClient()
        client.confirmConnection()        
        state_dim = len(get_distance_sensor_data(client, drone_name)) + DRONE_POSITION_LEN + TARGET_POSITION_LEN 
        env = AirsimDroneEnv(dqntools.calculate_reward, state_dim, client, drone_name, DISTANCE_SENSOR)
        agent = DQNAgent(state_dim=state_dim, action_dim=3)
        episodes = args.episodes

        objects = client.simListSceneObjects(f'{args.object}[\w]*')
        targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_BOTTOM_LIMIT)
        print(f'best path: {targets}')
        if len(targets) > 0:
            signal.signal(signal.SIGINT, signal_handler)
            if args.weight != '':
                try:
                    agent.load(args.weight)
                except:
                    print(f"The path:{args.weight} is not exist, load weight fail.")
            
            for episode in range(episodes):
                if stop_event.is_set(): # if stop event is set, stop training and save the weight
                    break
                client.reset()
                client.enableApiControl(True)
                time.sleep(0.5)
                targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_BOTTOM_LIMIT)
                state, _ = env.reset(targets[0])
                done = False
                rewards = 0
                step_count = 0
                while not done:
                    action = agent.act(state)
                    next_state, reward, done, _, info = env.step(action, targets = targets, drone_name=drone_name, step_cnt = step_count)                
                    
                    state = next_state
                    targets = info['targets']
                    rewards += reward # calculate total rewards
                    step_count += 1
                    if done:
                        if info['overlap']:
                            status = (f'Episode: {episode:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: fail')
                        else:
                            status = (f'Episode: {episode:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: success')
                    else:
                        status = (f'Episode: {episode:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: run')
                    sys.stdout.write('\r' + status)
                    sys.stdout.flush()

                print(f'\r')

            print("test finished!")
        else:
            print("The corresponding object cannot be found in the environment and testing cannot be started.")
