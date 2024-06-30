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