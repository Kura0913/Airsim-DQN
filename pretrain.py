from DQN.DQNAgent import DQNAgent
from DQN.Env import AirsimDroneEnv
import Tools.DQNTools as dqntools
import Tools.AirsimTools as airsimtools
import matplotlib.pyplot as plt
import airsim
import time
import numpy as np
import argparse
import torch
import os
import json
import time
import threading
import signal
import sys

ROUND_DECIMALS = 2
DRONE_POSITION_LEN = 3
TARGET_POSITION_LEN = 3
SPAWN_OBJECT_NAME = 'BP_spawn_point'
DISTANCE_SENSOR = ["front", "left", "right", "rfront", "lfront", "top", "bottom", 'lfbottom', 'rfbottom', 'lbbottom', 'rbbottom']

BASE_PTAH = '.\\runs\\pretrain\\'
# sensor name
DRONE_LIMIT = {
    'front':2.0,
    'right': 2.0,
    'left': 2.0,
    'bottom': 1.0,
    'top': 2.0
}

DRONE_MAX_SPEED = 3

DISTANCE_RANGE = (0, 2)
MAPING_RANGE = (0, 1)

OBJECT_NAME = "BP_Grid"

def get_distance_sensor_data(client:airsim.MultirotorClient, drone_name):
    sensor_data = []
    for sensor_name in DISTANCE_SENSOR:
        sensor_data.append(client.getDistanceSensorData(sensor_name, drone_name).distance)
    return sensor_data


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

def signal_handler(signum, frame):
    global stop_event
    global folder_path
    print("\nTraining interrupted. Saving model...")
    agent.save(f"{folder_path}\\model.pth")
    plot_rewards_and_losses(range(1, episode + 1), eposide_reward, eposide_loss_avg, save_path=f'{folder_path}\\final_performance_plot.png')
    print("Model saved. Exiting...")
    stop_event.set()
    sys.exit(0)

def drone_moving_state(client: airsim.MultirotorClient, target_position):
    # get dinstance sensor data
    front = client.getDistanceSensorData(DISTANCE_SENSOR[0]).distance
    rfront = client.getDistanceSensorData(DISTANCE_SENSOR[3]).distance
    lfront = client.getDistanceSensorData(DISTANCE_SENSOR[4]).distance
    right = client.getDistanceSensorData(DISTANCE_SENSOR[2]).distance
    left = client.getDistanceSensorData(DISTANCE_SENSOR[1]).distance
    top = client.getDistanceSensorData(DISTANCE_SENSOR[5]).distance
    bottom = client.getDistanceSensorData(DISTANCE_SENSOR[6]).distance
    rfbottom = client.getDistanceSensorData(DISTANCE_SENSOR[8]).distance
    lfbottom = client.getDistanceSensorData(DISTANCE_SENSOR[7]).distance
    rbbottom = client.getDistanceSensorData(DISTANCE_SENSOR[10]).distance
    lbbottom = client.getDistanceSensorData(DISTANCE_SENSOR[9]).distance

    drone_position = client.simGetVehiclePose().position

    velocity = airsimtools.get_velocity(drone_position, target_position, 0)
    RISE_VELOCITY = 1.0
    velocity_factor = 1.0

    if bottom < DRONE_LIMIT['bottom'] and top > DRONE_LIMIT['top']:
        correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, bottom)]
        stop_action = True
    elif rfbottom < DRONE_LIMIT['bottom'] and top > DRONE_LIMIT['top']:
        correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, rfbottom)]
        stop_action = True
    elif lfbottom < DRONE_LIMIT['bottom'] and top > DRONE_LIMIT['top']:
        correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, lfbottom)]
        stop_action = True
    elif rbbottom < DRONE_LIMIT['bottom'] and top > DRONE_LIMIT['top']:
        correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, rbbottom)]
        stop_action = True
    elif lbbottom < DRONE_LIMIT['bottom'] and top > DRONE_LIMIT['top']:
        correct_velocity = [0, 0, -1 + airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, lbbottom)]
        stop_action = True
    elif top < DRONE_LIMIT['top'] and bottom > DRONE_LIMIT['bottom'] and rfbottom > DRONE_LIMIT['bottom'] and lfbottom > DRONE_LIMIT['bottom'] and rbbottom > DRONE_LIMIT['bottom'] and lbbottom > DRONE_LIMIT['bottom']:
        correct_velocity = [0, 0, 1 - airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, lbbottom)]
        stop_action = True
    else:
        if front < DRONE_LIMIT['front'] and left < DRONE_LIMIT['left'] and right < DRONE_LIMIT['right']:# sensor: front, left, right on
            correct_velocity = [0, 0, -1]
            stop_action = True
        elif front < DRONE_LIMIT['front'] and left < DRONE_LIMIT['left']:# sensor: front, left on
            correct_velocity = [0, 1, 0]
            stop_action = True
        elif front < DRONE_LIMIT['front'] and right < DRONE_LIMIT['right']:# sensor: front, right on
            correct_velocity = [0, -1, 0]
            stop_action = True
        elif left < DRONE_LIMIT['left']:# sensor: left on
            correct_velocity = [0, 1, 0]
            velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, left)
            stop_action = False
        elif right < DRONE_LIMIT['right']:# sensor: right on
            correct_velocity = [0, -1, 0]
            velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, right)
            stop_action = False
        elif front < DRONE_LIMIT['front']:# senso: front on
            if rfront > lfront:# there is more space at right side
                correct_velocity = [0, 1, 0]
                velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, right)
                stop_action = False
            else:# there is more space at left side
                correct_velocity = [0, -1, 0]
                velocity_factor = airsimtools.map_value(DISTANCE_RANGE, MAPING_RANGE, left)
                stop_action = False
        else:# normal
            correct_velocity = [0, 0, 0]
            stop_action = False


    if stop_action: # drone is very close to obstacles, need to stop moving
        return correct_velocity
    else:# there is enough space to correct the velocity
        velocity = [i * velocity_factor for i in velocity]
        correct_velocity[2] -= (RISE_VELOCITY - velocity_factor)
        return np.sum([velocity, correct_velocity], axis=0).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AirSim-DQN train.")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--episodes', type=int, default=5, help='number of training')
    parser.add_argument('--gamma', type=float, default=0.99, help='weight of future reward')
    parser.add_argument('--infinite_loop', type=bool, default=False, help='keep training until press the stop button')
    parser.add_argument('--weight', type=str, default='', help='weight path')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    parser.add_argument('--object', type=str, default='BP_Grid', help='The object name in the vr environment, you can place objects in the VR environment and make sure that the objects you want to visit start with the same name.. Initial object is: BP_Grid')
    args = parser.parse_args()
    
    # to stop training and save the weight
    stop_event = threading.Event()
    client = airsim.MultirotorClient()
    client.confirmConnection()

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
        agent = DQNAgent(state_dim=state_dim, action_dim=3, bacth_size=args.batch_size, gamma=args.gamma, device=device)
        episodes = args.episodes
        objects = client.simListSceneObjects(f'{args.object}[\w]*')        
        targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_LIMIT['bottom'])
        spwan_objects = client.simListSceneObjects(f'{SPAWN_OBJECT_NAME}[\w]*')
        spawn_points = airsimtools.get_targets(client, spwan_objects, ROUND_DECIMALS, DRONE_LIMIT['bottom'])
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
                targets = airsimtools.get_targets(client, objects, ROUND_DECIMALS, DRONE_LIMIT['bottom'])
                state, _ = env.reset(targets[0])
                done = False
                rewards = 0
                step_count = 0
                total_loss = 0
                agent.train_cnt = 0
                while not done:
                    action = airsimtools.scale_and_normalize_vector(drone_moving_state(client, targets[0]), 1)
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
    