import os
import random

data_path = '/home/fyp3-2/Desktop/BATCH18/Weligama_data'
image_folder_path = os.path.join(data_path, 'images')

boat_folders = []

for root, dirs, files in os.walk(data_path, topdown=True):
    boat_folders = dirs
    break

new_boat_names = []
for i, boat in enumerate(boat_folders):
    num = f'{i:03d}'
    os.rename(os.path.join(data_path, boat), os.path.join(data_path, num))
    new_boat_names.append(num)

train_split = 0.2
random.shuffle(new_boat_names)
index_range = int(len(new_boat_names) * train_split)
train_boats = new_boat_names[index_range + 1:]
test_boats = new_boat_names[:index_range]

os.mkdir('/home/fyp3-2/Desktop/BATCH18/Weligama_data/reid/train')
for boat in new_boat_names:

