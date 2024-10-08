from DQN.DQNAgent import DQNAgent
from DQN.Env import AirsimDroneEnv
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
import time
import signal

ROUND_DECIMALS = 2
DRONE_BOTTOM_LIMIT = 1
DRONE_POSITION_LEN = 3
TARGET_POSITION_LEN = 3
SPAWN_OBJECT_NAME = 'BP_spawn_point'

DISTANCE_SENSOR = ["front", "left", "right", "rfront", "lfront", "top", "bottom", 'lfbottom', 'rfbottom', 'lbbottom', 'rbbottom']

BASE_PTAH = '.\\runs\\train\\'

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    sensor_data = []
    for sensor_name in DISTANCE_SENSOR:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data

def signal_handler(signum, frame):
    global stop_event
    global folder_path
    print("\nTraining interrupted. Saving model...")
    agent.save(f"{folder_path}\\model.pth")
    plot_rewards_and_losses(range(1, episode + 1), eposide_reward, eposide_loss_avg, save_path=f'{folder_path}\\final_performance_plot.png')
    print("Model saved. Exiting...")
    stop_event.set()
    sys.exit(0)

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
    parser = argparse.ArgumentParser(description="AirSim-DQN train.")
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
        # get weight save folder path
        folder_path = dqntools.create_directory(BASE_PTAH)
        drone_name = vehicle_names[0]
        client = airsim.MultirotorClient()
        client.confirmConnection()
        # len(get_distance_sensor_data(client, drone_name)) + DRONE_POSITION_LEN + TARGET_POSITION_LEN
        state_dim = len(get_distance_sensor_data(client, drone_name)) + DRONE_POSITION_LEN + TARGET_POSITION_LEN 
        env = AirsimDroneEnv(dqntools.calculate_reward, state_dim, client, drone_name, DISTANCE_SENSOR)
        agent = DQNAgent(state_dim=state_dim, action_dim=3, bacth_size=args.batch_size, epsilon=args.epsilon, decay_episode=args.decay_episode, gamma=args.gamma, device=device)
        episodes = args.episodes

        objects = client.simListSceneObjects(f'{args.object}[\w]*')        
        targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_BOTTOM_LIMIT)
        spwan_objects = client.simListSceneObjects(f'{SPAWN_OBJECT_NAME}[\w]*')
        spawn_points = airsimtools.get_targets(client, spwan_objects, ROUND_DECIMALS, DRONE_BOTTOM_LIMIT)
        print('best path:', targets)

        if len(targets) > 0:
            if args.weight != '':
                try:
                    agent.load(args.weight)
                except:
                    print(f"The path:{args.weight} is not exist, load weight fail.")

            signal.signal(signal.SIGINT, signal_handler)
            episode = 0
            eposide_reward = []
            eposide_loss_avg = []
            while episode < episodes:
                if stop_event.is_set(): # if stop event is set, stop training and save the weight
                    break
                airsimtools.reset_drone_to_random_spawn_point(client, drone_name, spawn_points)
                time.sleep(1)
                targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_BOTTOM_LIMIT)
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
            agent.save(f"{folder_path}\\model.pth") # save weight
            plot_rewards_and_losses(range(1, episode + 1), eposide_reward, eposide_loss_avg, save_path=f'{folder_path}\\final_performance_plot.png')
            print("Updated model saved!")
        else:
            print("The corresponding object cannot be found in the environment and training cannot be started.")
