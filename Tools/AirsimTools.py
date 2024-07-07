import numpy as np
import math
import airsim
from ShortestPath import TravelerShortestPath as tsp

# check the value wheather equals to "negative zero",if yes, set them to 0.0
def check_negative_zero(x, y, z):
    if x == -0.0:
        x = float('0.0')
    if y == -0.0:
        y = float('0.0')
    if z == -0.0:
        z = float('0.0')

    return [x, y, z]

def calculate_distance(coord1, coord2):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# get normalized vector * n
def scale_and_normalize_vector(v, n):
    # get vector length
    magnitude = np.linalg.norm(v)
    
    # normalized
    if magnitude != 0:
        unit_vector = v / magnitude
    else:
        unit_vector = v

    scaled_vector = n * unit_vector
    return scaled_vector

# get the horizontal rotation angle
def calculate_horizontal_rotation_angle(velocity):
    horizontal_projection = [velocity[0], velocity[1]]

    angle = math.atan2(horizontal_projection[1], horizontal_projection[0])

    angle_in_degree = math.degrees(angle)
    return angle_in_degree

def get_yaw_mode_F(velocity):
    angle_in_degree = calculate_horizontal_rotation_angle(velocity)

    return airsim.YawMode(False, angle_in_degree)

def get_velocity(vehicle_position, target_position, round_decimals: int):
    ### variable type:
    ### vehicle_position: airsim.Pose.position
    vehicle_position = check_negative_zero(np.round(vehicle_position.x_val, round_decimals), np.round(vehicle_position.y_val, round_decimals), np.round(vehicle_position.z_val, round_decimals))
    velocity = [x - y for x, y in zip(target_position, vehicle_position)]

    return velocity

def map_value(value_range: tuple, target_range: tuple, value: float):
    # get value range
    min_value, max_value = value_range
    target_min, target_max = target_range

    # get ratio
    ratio = (value - min_value) / (max_value - min_value)

    # mapping the value
    mapped_value = target_min + ratio * (target_max - target_min)

    if mapped_value > target_max:
        return target_max
    elif mapped_value < target_min:
        return target_min
    else:
        return mapped_value

def get_targets(client, targets, round_decimals, bottom_limit):

    target_pos_ary = []

    for target in targets:
        target_pos = client.simGetObjectPose(target).position
        target_pos = [np.round(target_pos.x_val, round_decimals), np.round(target_pos.y_val, round_decimals), np.round(target_pos.z_val, round_decimals)]
        target_pos = check_negative_zero(target_pos[0], target_pos[1], target_pos[2] - bottom_limit)
        target_pos_ary.append(target_pos)
    
    drone_pos = client.simGetVehiclePose().position
    drone_pos = check_negative_zero(np.round(drone_pos.x_val, round_decimals), np.round(drone_pos.y_val, round_decimals), np.round(drone_pos.z_val, round_decimals))
    target_pos_ary = tsp.getTSP(target_pos_ary, drone_pos)
    del target_pos_ary[0]
    return target_pos_ary