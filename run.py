from DQN.DQNAgent import DQNAgent
from DQN.Env import AirsimDroneEnv
from ShortestPath import TravelerShortestPath as tsp
import Tools.AirsimTools as airsimtools
import Tools.DQNTools as dqntools
import matplotlib.pyplot as plt
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
DRONE_POSITION_LEN = 3
TARGET_POSITION_LEN = 3
DISTANCE_SENSOR = ["front", "left", "right", "rfront", "lfront", "top", "bottom", 'lfbottom', 'rfbottom', 'lbbottom', 'rbbottom']

BASE_PTAH = '.\\runs\\'

exit_flag = False

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    sensor_data = []
    for sensor_name in DISTANCE_SENSOR:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data

def calculate_reward(done, overlap, prev_dis, curr_dis, targets):
    arrive_target = False
    if curr_dis < 0.5:
            arrive_target = True
            del targets[0]
    if done:
        if overlap: # mission fail
            return -10, {'prev_dis' : -1, 'targets': targets}
        else: # mission success
            return 15, {'prev_dis' : -1, 'targets': targets}
    else:
        if arrive_target: # arrive current target, reset the distance value to -1
            return 5, {'prev_dis' : -1, 'targets': targets}
        else:
            if prev_dis < 0 or curr_dis < prev_dis: # drone is closer the target than before
                return 1, {'prev_dis' : curr_dis, 'targets': targets}
            else: # drone is further the target than before
                return -1, {'prev_dis' : curr_dis, 'targets': targets}

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
    return target_pos_ary

# waiting for pressing 'p' key to stop
def listen_for_stop():  
    global exit_flag  
    while not exit_flag:
        if keyboard.is_pressed('p'):
            stop_event.set()
            break
        time.sleep(0.1)

def plot_rewards_and_losses(episodes, rewards, average_losses, save_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot rewards as bars
    ax1.bar(episodes, rewards, color='blue', alpha=0.6, label='Rewards')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Rewards', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for average losses
    ax2 = ax1.twinx()
    ax2.plot(episodes, average_losses, color='red', label='Average Loss')
    ax2.set_ylabel('Average Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legends and grid
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    ax1.grid(True)

    # Save and show the plot
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distance between two coordinates.")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--episodes', type=int, default=5, help='number of training')
    parser.add_argument('--decay_episode', type=int, default=500, help='set the episode where epsilon starts to decay')
    parser.add_argument('--gamma', type=float, default=0.99, help='weight of future reward')
    parser.add_argument('--epsilon', type=float, default=1, help='random action rate')
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
        # len(get_distance_sensor_data(client, drone_name)) + DRONE_POSITION_LEN + TARGET_POSITION_LEN
        state_dim = len(get_distance_sensor_data(client, drone_name)) + DRONE_POSITION_LEN + TARGET_POSITION_LEN 
        env = AirsimDroneEnv(calculate_reward, state_dim, client, drone_name, DISTANCE_SENSOR)
        agent = DQNAgent(state_dim=state_dim, action_dim=3, bacth_size=args.batch_size, epsilon=args.epsilon, decay_episode=args.decay_episode, gamma=args.gamma, device=device)
        episodes = args.episodes

        objects = client.simListSceneObjects(f'{args.object}[\w]*')
        targets = get_targets(client, objects, ROUND_DECIMALS)
        print('best path:', targets)
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
            eposide_reward = []
            eposide_loss_avg = []
            while episode < episodes:
                if stop_event.is_set(): # if stop event is set, stop training and save the weight
                    break
                client.reset()
                client.enableApiControl(True)
                state, _ = env.reset(targets[0])
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
                    
                    loss, curr_epsilon = agent.train(episode)
                    curr_epsilon = np.round(curr_epsilon, 4)
                    if loss >= 0:
                        total_loss += loss
                    rewards += reward # calculate total rewards
                    step_count += 1
                    if agent.train_cnt == 0:
                        loss_avg = 0
                    else:
                        loss_avg = np.round(total_loss.cpu().detach().numpy() / agent.train_cnt, 4)
                    if args.infinite_loop:
                        if done:
                            targets = get_targets(client, objects, ROUND_DECIMALS)
                            if info['overlap']:
                                status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: fail')
                            else:
                                status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: success')
                        else:
                            status = (f'Episode: {episode + 1:5d}/N | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: run')
                    else:
                        if done:
                            if info['overlap']:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: fail')
                            else:
                                status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: success')
                        else:
                            status = (f'Episode: {episode + 1:5d}/{episodes} | Step: {step_count:3d} | Reward: {rewards:5d} | loss: {loss_avg:.4f} | epsilon: {curr_epsilon:.4f} | mission_state: run')
                        
                    sys.stdout.write('\r' + status)
                    sys.stdout.flush()
                print(f'\r')
                eposide_reward.append(rewards)
                eposide_loss_avg.append(loss_avg)
                if not args.infinite_loop:
                    episode += 1
            folder_path = dqntools.create_directory(BASE_PTAH)
            agent.save(f"{folder_path}\\model.pth") # save weight
            plot_rewards_and_losses(range(1, episode + 1), eposide_reward, eposide_loss_avg, save_path=f'{folder_path}\\final_performance_plot.png')
            print("Updated model saved!")
            exit_flag = True
            stop_thread.join()
        else:
            print("The corresponding object cannot be found in the environment and training cannot be started.")
