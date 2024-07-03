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
import torch
import threading
import keyboard
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

ROUND_DECIMALS = 2
BASE_PTAH = '.\\runs\\'

exit_flag = False

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
    curr_distance = airsimtools.calculate_distance(drone_position, curr_target)
    if done:
        if overlap:
            return -1000, {'prev_dis' : -1}
        else:
            return 1000, {'prev_dis' : -1}
    else:
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

# waiting for pressing 'p' key to stop
def listen_for_stop():  
    global exit_flag  
    while not exit_flag:
        if keyboard.is_pressed('p'):
            stop_event.set()
            break
        time.sleep(0.1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distance between two coordinates.")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--episodes', type=int, default=5, help='number of training')
    parser.add_argument('--gamma', type=float, default=0.99, help='weight of previous reward')
    parser.add_argument('--infinite_loop', type=bool, default=False, help='keep training until press the stop button')
    parser.add_argument('--weight', type=str, default='', help='weight path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    parser.add_argument('--object', type=str, default='BP_Grid', help='The object name in the vr environment, you can place objects in the VR environment and make sure that the objects you want to visit start with the same name.. Initial object is: BP_Grid')
    args = parser.parse_args()
    # to stop training and save the weight    
    stop_event = threading.Event()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
        agent = DQNAgent(state_dim=sensor_num, action_dim=4, bacth_size=args.batch_size, gamma=args.gamma, device=device)
        episodes = args.episodes

        objects = client.simListSceneObjects(f'{args.object}[\w]*')
        targets = get_targets(client, objects, ROUND_DECIMALS)
        # start the thread
        stop_thread = threading.Thread(target=listen_for_stop)
        stop_thread.start()

        if len(targets) > 0:
            if args.weight != '':
                try:
                    agent.load(args.weight)
                except:
                    print(f"The path:{args.weight} is not exist, load weight fail.")
            
            infinit_loop_cnt = 0
            for episode in range(episodes):
                if stop_event.is_set(): # if stop event is set, stop training and save the weight
                    break
                client.reset()
                client.enableApiControl(True)
                state, _ = env.reset()
                done = False
                rewards = 0
                step_count = 0
                infinit_loop_cnt += 1
                while not done:
                    action = agent.act(state)
                    n, e, d, alpha = action
                    n, e, d = airsimtools.scale_and_normalize_vector([n, e, d], 1)
                    if step_count <= 0:
                        client.takeoffAsync(10, drone_name).join()
                    
                    drone_pos = client.simGetVehiclePose(drone_name).position
                    velocity = airsimtools.scale_and_normalize_vector(airsimtools.get_velocity(drone_pos, targets[0], 2), 1)
                    velocity = [i  * alpha / 100 for i in velocity]
                    drone_pos = airsimtools.check_negative_zero(np.round(drone_pos.x_val, ROUND_DECIMALS), np.round(drone_pos.y_val, ROUND_DECIMALS), np.round(drone_pos.z_val, ROUND_DECIMALS))

                    client.moveByVelocityAsync(velocity[0] + float(n),velocity[1] +  float(e),velocity[2] +  float(d) , 0.1).join()
                    
                    sensor_values = get_distance_sensor_data(client, drone_name)
                    next_state, reward, done, _, info = env.step(action, sensor_values, targets = targets, drone_position=drone_pos, step_cnt = step_count)                
                    agent.store_experience(state, action, reward, next_state, done)
                    state = next_state
                    
                    agent.train()
                    rewards += reward # calculate total rewards
                    step_count += 1
                    if args.infinite_loop:
                        episode -= 1
                        if done:
                            if info['overlap']:
                                status = (f'Episode: {infinit_loop_cnt:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: fail')
                            else:
                                status = (f'Episode: {infinit_loop_cnt:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: success')
                        else:
                            status = (f'Episode: {infinit_loop_cnt:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: run')
                    else:
                        if done:
                            if info['overlap']:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: fail')
                            else:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: success')
                        else:
                            status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | mission_state: run')
                    sys.stdout.write('\r' + status)
                    sys.stdout.flush()

                print(f'\r')

            agent.save(f"{dqntools.create_directory(BASE_PTAH)}\\model.pth") # save weight
            print("Updated model saved!")
            exit_flag = True
            stop_thread.join()
        else:
            print("The corresponding object cannot be found in the environment and training cannot be started.")
