import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("/home/fyp3-2/Desktop/BATCH18/ReID_check/temp.pth")
if torch.cuda.is_available():
    model.cuda()


def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size = 1024
    part_feature_size = 512
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    distance = (x1 - x2).pow(2)
    distance[:global_feature_size] = distance[:global_feature_size] * normalized_cam[0:1]
    distance[global_feature_size: global_feature_size + part_feature_size] = distance[
                                                                             global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                                                             1:2]
    distance[global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] = distance[
                                                                                                     global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                                                                             2:3]
    distance[global_feature_size + 2 * part_feature_size:] = distance[
                                                             global_feature_size + 2 * part_feature_size:] * normalized_cam[
                                                                                                             3:]

    return torch.sum(distance).item()


def get_area_ratios(image_name, mask_root):
    image = os.path.join(mask_root, image_name)
    front = Image.open(image.replace('.jpg', '_front.jpg'))
    front_area = np.sum(np.array(front) / 255)
    rear = Image.open(image.replace('.jpg', '_rear.jpg'))
    rear_area = np.sum(np.array(rear) / 255)
    side = Image.open(image.replace('.jpg', '_side.jpg'))
    side_area = np.sum(np.array(side) / 255)
    global_area = front_area + rear_area + side_area
    front_area /= global_area
    rear_area /= global_area
    side_area /= global_area
    global_area /= global_area
    area_ratios = np.array([global_area, front_area, rear_area, side_area])
    return area_ratios



def compare(query_mask_dir, gallery_mask_dir, query_image, query_img_features, gallery_image, gallery_img_features):
    query_area_ratios = get_area_ratios(query_image, query_mask_dir)
    gallery_area_ratios = get_area_ratios(gallery_image, gallery_mask_dir)

    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    return weighted_distance


def accuracy(query_mask_dir, gallery_mask_dir, query_images, query_images_ids, query_images_features, gallery_images, gallery_images_ids, gallery_images_features):

    pbar = tqdm(total=len(query_images))
    distances = {}
    for j in range(len(gallery_images)):
        weighted_distance = compare(query_mask_dir, gallery_mask_dir, query_images[i], query_images_features[i], gallery_images[j], gallery_images_features[j])
        distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])


