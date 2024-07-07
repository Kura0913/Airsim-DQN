import os
from datetime import datetime

def create_directory(base_path):
    current_date = datetime.now().strftime("%Y%m%d")
    
    folder_number = 1
    
    while True:
        folder_name = f"{current_date}{folder_number:02d}"
        directory_path = os.path.join(base_path, folder_name)
        
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return directory_path
        
        folder_number += 1

def calculate_reward(done, overlap, prev_dis, curr_dis, arrive_target):
        
    if done:
        if overlap: # mission fail
            return -10, {'prev_dis' : -1}
        else: # mission success
            return 15, {'prev_dis' : -1}
    else:
        if arrive_target: # arrive current target, reset the distance value to -1
            return 5, {'prev_dis' : -1}
        else:
            if prev_dis < 0 or curr_dis < prev_dis: # drone is closer the target than before
                return 1, {'prev_dis' : curr_dis}
            else: # drone is further the target than before
                return -1, {'prev_dis' : curr_dis}