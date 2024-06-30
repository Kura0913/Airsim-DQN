import numpy as np
from itertools import permutations


def heuristicEstimateOfDistance(start, goal):
    return np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2 + (start[2] - goal[2])**2)

def reconstructPath(came_from,current_node):
    p = []
    # print(current_node)
    if str(current_node) in came_from:
        # print('exist')
        for each in reconstructPath(came_from,came_from[str(current_node)]):
            p.append(each)
        p.append(current_node)
        # print(p)

        return p
    else:
        # print('not exist')
        p.append(current_node)
        return p

class TravelerShortestPath():
    def getTSP(coordinates, start_coordinate):
        # add drone position to coordinates list
        coordinates_with_start = [start_coordinate] + coordinates

        n = len(coordinates_with_start)
        cost_matrix = [[0] * n for _ in range(n)]

        # calculate the distance between each pair of points as a cost matrix
        for i in range(n):
            for j in range(n):
                cost_matrix[i][j] = sum((a - b)**2 for a, b in zip(coordinates_with_start[i], coordinates_with_start[j]))**0.5

        # find the index of the starting coordinate in coordinates_with_start
        start_index = coordinates_with_start.index(start_coordinate)

        # initial setting
        min_length = float('inf')
        optimal_path_indices = None

        # Get all possible paths except the drone position
        all_paths = permutations(range(1, n))

        # calculate the total length of all paths
        for path in all_paths:
            # prefix the path with the index of the starting coordinates
            path_indices = (0,) + path
            total_length = sum(cost_matrix[i][j] for i, j in zip(path_indices, path_indices[1:]))
            
            # if a shorter path is found, update the minimum length and path index
            if total_length < min_length:
                min_length = total_length
                optimal_path_indices = path_indices

        # reorder according to the index of the shortest path
        path_order = [coordinates_with_start[i] for i in optimal_path_indices]

        return path_order
