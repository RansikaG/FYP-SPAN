import os
import random
import shutil

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

os.mkdir(data_path + '/reid/train')
os.mkdir(data_path + '/reid/valid')
for boat in train_boats:
    path = os.path.join(data_path, boat)
    for (root, dirs, files) in os.walk(path, topdown=True):
        images = files
        random_img = random.choice(images)
        valid_image_path = data_path + '/reid/valid/' + boat
        os.mkdir(valid_image_path)
        shutil.move(os.path.join(root, random_img), os.path.join(valid_image_path, random_img))
        train_image_path = data_path + '/reid/train'
        shutil.move(os.path.join(root, boat), os.path.join(train_image_path, boat))

os.mkdir(data_path + '/reid/gallery')
os.mkdir(data_path + '/reid/query')
for boat in test_boats:
    path = os.path.join(image_folder_path, boat)
    for (root, dirs, files) in os.walk(path, topdown=True):
        images = files
        random_img = random.choice(images)
        query_image_path = data_path + '/reid/query/' + boat
        os.mkdir(query_image_path)
        shutil.move(os.path.join(root, random_img), os.path.join(query_image_path, random_img))
        gallery_image_path = data_path + '/reid/gallery'
        shutil.move(os.path.join(root, boat), os.path.join(gallery_image_path, boat))
