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

ROUND_DECIMALS = 2
DISTANCE_SENSOR = ["front", "left", "right", "rfront", "lfront", "top", "bottom", 'lfbottom', 'rfbottom', 'lbbottom', 'rbbottom']

BASE_PTAH = '.\\runs\\'

exit_flag = False

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    sensor_data = []
    for sensor_name in DISTANCE_SENSOR:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data

def calculate_reward(done, overlap, prev_dis, curr_dis, targets):
    reward = 0
    if curr_dis < 0.5:
            reward += 5
            del targets[0]
    if done:
        if overlap: # mission fail
            return reward - 10, {'prev_dis' : -1, 'targets': targets}
        else: # mission success
            return reward + 10, {'prev_dis' : -1, 'targets': targets}
    else:
        if reward > 0: # arrive current target, reset the distance value to -1
            if prev_dis < 0 or curr_dis < prev_dis:
                return reward + 1, {'prev_dis' : -1, 'targets': targets}
            else:
                return reward - 2, {'prev_dis' : -1, 'targets': targets}
        else:
            if prev_dis < 0 or curr_dis < prev_dis: # drone is closer the target than before
                return reward + 1, {'prev_dis' : curr_dis, 'targets': targets}
            else: # drone is further the target than before
                return reward - 2, {'prev_dis' : curr_dis, 'targets': targets}

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
    parser.add_argument('--episodes', type=int, default=10000, help='number of training')
    parser.add_argument('--epsilon', type=float, default=0.99, help='weight of previous reward')
    parser.add_argument('--epsilon_min', type=float, default=0.2, help='epsilon\'s minimum')
    parser.add_argument('--decay', type=float, default=0.999, help='epsilon\'s decay rate')
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
        state_dim = len(get_distance_sensor_data(client, drone_name)) + 3
        env = AirsimDroneEnv(calculate_reward, state_dim, client, DISTANCE_SENSOR)
        agent = DQNAgent(state_dim=state_dim, action_dim=3, bacth_size=args.batch_size, epsilon=args.epsilon, device=device)
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
            
            episode = 0
            while episode < episodes:
                if stop_event.is_set(): # if stop event is set, stop training and save the weight
                    break
                client.reset()
                client.enableApiControl(True)
                state, _ = env.reset()
                done = False
                rewards = 0
                step_count = 0
                total_loss = 0
                agent.train_cnt = 0
                while not done:
                    action = agent.act(state)                    
                    next_state, reward, done, _, info = env.step(action, targets, step_cnt=step_count, drone_name=drone_name)
                    agent.store_experience(state, action, reward, next_state, done)
                    state = next_state
                    targets = info['targets']
                    
                    loss = agent.train()
                    if loss >= 0:
                        total_loss += loss
                    rewards += reward # calculate total rewards
                    step_count += 1
                    if agent.train_cnt == 0:
                        loss_avg = 0
                    else:
                        loss_avg = np.round(total_loss.cpu().detach().numpy() / agent.train_cnt, 4)
                    if args.infinite_loop:
                        episode -= 1
                        if done:
                            targets = get_targets(client, objects, ROUND_DECIMALS)
                            if info['overlap']:
                                status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: fail')
                            else:
                                status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: success')
                        else:
                            status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: run')
                    else:
                        if done:
                            if info['overlap']:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: fail')
                            else:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: success')
                        else:
                            status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | mission_state: run')
                    sys.stdout.write('\r' + status)
                    sys.stdout.flush()
                episode += 1
                print(f'\r')

            agent.save(f"{dqntools.create_directory(BASE_PTAH)}\\model.pth") # save weight
            print("Updated model saved!")
            exit_flag = True
            stop_thread.join()
        else:
            print("The corresponding object cannot be found in the environment and training cannot be started.")
