import torch
import numpy as np
#import time
import os
from PIL import Image
import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.Second_Stage_Extractor()

def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size=1024
    part_feature_size=512
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam, axis=1, keepdims=True)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    distance = (x1 - x2).pow(2)
    global_distance = distance[:, :global_feature_size] * normalized_cam[:, 0:1]
    front_distance = distance[:,
                     global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                    :, 1:2]
    rear_distance = distance[:,
                    global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                :,
                                                                                                                                2:3]
    side_distance = distance[:, global_feature_size + 2 * part_feature_size:] * normalized_cam[:, 3:]
    weighted_distance = torch.cat((global_distance, front_distance, rear_distance, side_distance), 1).sum(1)
    return weighted_distance


def get_area_ratios(image_name):
    # image_root='./PartAttMask/image_query'
    # image = os.path.join(image_root, image_name)
    front = Image.open(image_name.replace('.jpg', '_front.jpg'))
    front_area = np.sum(np.array(front) / 255)
    rear = Image.open(image_name.replace('.jpg', '_rear.jpg'))
    rear_area = np.sum(np.array(rear) / 255)
    side = Image.open(image_name.replace('.jpg', '_side.jpg'))
    side_area = np.sum(np.array(side) / 255)
    global_area = front_area + rear_area + side_area
    front_area /= global_area
    rear_area /= global_area
    side_area /= global_area
    global_area /= global_area
    # print('global: {} \nfront: {} \nrear:{} \nside: {}'.format(global_area, front_area, rear_area, side_area))
    area_ratios = np.array([global_area, front_area, rear_area, side_area])
    return area_ratios

def compare(query_image, gallery_image):
    query_img_features = model(query_image)
    gallery_img_features = model(gallery_image)
    query_area_ratios = calc_euclidean(query_image)
    gallery_area_ratios = calc_euclidean(gallery_image)

    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    
    return weighted_distance

def accuracy(query_images, query_images_ids, gallery_images, gallery_images_ids, top):
    final_accuracy = 0
    for i in range(len(query_images)):
        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_images[i], gallery_images[j])
            distances[gallery_images_ids[j]] = weighted_distance
        sorted_distances = sorted(distances.items(), key=lambda x:x[1])
        correct_instances = sorted_distances.keys()[:top].count(query_images_ids[i])
        final_accuracy += correct_instances
    return final_accuracy/(top*len(query_images))


def mAP(query_images, query_images_id, num_of_ids, gallery_images, gallery_images_ids):
    total = 0
    for i in range(len(query_images)):
        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_images[i], gallery_images[j])
            distances[gallery_images_ids[j]] = weighted_distance
        sorted_distances = sorted(distances.items(), key=lambda x:x[1])
        instances = sorted_distances.keys()
        correct_instances = 0
        precision_total = 0
        for x in range(1, len(instances) +  1):
            if correct_instances == num_of_ids[i]:
                break
            elif instances[x] == query_images_id[i]:
                correct_instances += 1
                precision_total  += correct_instances / x
        total += precision_total / num_of_ids[i]
    return total / len(query_images)


top1 = accuracy(query_images, query_images_id, gallery_images, gallery_images_ids, 1)
top5 = accuracy(query_images, query_images_id, gallery_images, gallery_images_ids, 5)
top10 = accuracy(query_images, query_images_id, gallery_images, gallery_images_ids, 10)
MAP = mAP(query_images, query_images_id, num_of_ids, gallery_images, gallery_images_ids)

print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(top1,top5,top10,MAP))