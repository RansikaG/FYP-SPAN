import torch
import numpy as np
# import time
import os
import torch.nn as nn
import pathlib
from PIL import Image
# import model
import cv2
from torchvision.transforms import transforms

import random

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

    # weighted_distance = [global_distance, front_distance, rear_distance, side_distance]
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


def compare(query_image, query_image_id, gallery_image, gallery_image_id):
    img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

    imageQueryT = img_transform(Image.open(
        '/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/query/' + query_image_id + '/' + query_image).convert('RGB'))
    imageQuery = torch.unsqueeze(imageQueryT, 0)
    frontQueryT = mask_transform(
        Image.open('/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/query_masks/' + query_image[:-4] + '_front.jpg'))
    frontQuery = torch.unsqueeze(frontQueryT, 0)
    rearQueryT = mask_transform(
        Image.open('/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/query_masks/' + query_image[:-4] + '_rear.jpg'))
    rearQuery = torch.unsqueeze(rearQueryT, 0)
    sideQueryT = mask_transform(
        Image.open('/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/query_masks/' + query_image[:-4] + '_side.jpg'))
    sideQuery = torch.unsqueeze(sideQueryT, 0)

    imageGalleryT = img_transform(Image.open(
        '/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/gallery/' + gallery_image_id + '/' + gallery_image).convert(
        'RGB'))
    imageGallery = torch.unsqueeze(imageGalleryT, 0)
    frontGalleryT = mask_transform(Image.open(
        '/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/gallery_masks/' + gallery_image[:-4] + '_front.jpg'))
    frontGallery = torch.unsqueeze(frontGalleryT, 0)
    rearGalleryT = mask_transform(
        Image.open('/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/gallery_masks/' + gallery_image[:-4] + '_rear.jpg'))
    rearGallery = torch.unsqueeze(rearGalleryT, 0)
    sideGalleryT = mask_transform(
        Image.open('/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/gallery_masks/' + gallery_image[:-4] + '_side.jpg'))
    sideGallery = torch.unsqueeze(sideGalleryT, 0)

    # print(imageQuery.shape)
    query_img_features = model(imageQuery.to(device), frontQuery.to(device), rearQuery.to(device), sideQuery.to(device))
    gallery_img_features = model(imageGallery.to(device), frontGallery.to(device), rearGallery.to(device),
                                 sideGallery.to(device))

    query_area_ratios = get_area_ratios(query_image, "/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/query_masks")
    gallery_area_ratios = get_area_ratios(gallery_image,
                                          "/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula/gallery_masks")

    # weighted_distance = calc_euclidean(np.array([2,3,3,4,6]), query_area_ratios, np.array([1,4,5,7,8]), gallery_area_ratios)
    #
    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    # print(weighted_distance)
    # num1 = random.randint(0, 99)

    # return
    # return num1
    return weighted_distance


def accuracy(query_images, query_images_ids, gallery_images, gallery_images_ids):
    # divider = "###########"
    # print(query_images)
    # print(divider)
    # print(query_images_ids)
    # print(divider)
    # print(gallery_images)
    # print(divider)
    # print(gallery_images_ids)
    # print(divider)
    final_accuracy1 = 0
    final_accuracy5 = 0
    final_accuracy10 = 0
    for i in range(len(query_images)):
        # print("step" + str(i+1))
        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_images[i], query_images_ids[i], gallery_images[j], gallery_images_ids[j])
            distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
        # print(distances)
        # print(divider)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        # print(sorted_distances)
        keys = []
        # print(divider)
        for k in sorted_distances:
            keys.append(k[0][:3])
        # print(keys)
        correct_instances1 = keys[:1].count(query_images_ids[i])
        if keys[:5].count(query_images_ids[i]) >= 1:
            correct_instances5 = 1
        else:
            correct_instances5 = 0
        if keys[:10].count(query_images_ids[i]) >= 1:
            correct_instances10 = 1
        else:
            correct_instances10 = 0
        # print("step" + str(i+1), correct_instances)
        final_accuracy1 += correct_instances1
        final_accuracy5 += correct_instances5
        final_accuracy10 += correct_instances10
    return final_accuracy1 / len(query_images), final_accuracy5 / len(query_images), final_accuracy10 / len(
        query_images)


def mAP(query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids):
    total = 0
    for i in range(len(query_images)):
        # print("step" + str(i+1))
        distances = {}
        for j in range(len(gallery_images)):
            weighted_distance = compare(query_images[i], query_images_ids[i], gallery_images[j], gallery_images_ids[j])
            distances[gallery_images_ids[j] + gallery_images[j]] = weighted_distance
        # print(distances)
        # print(divider)
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        # print(sorted_distances)
        # print(divider)
        instances = []
        for k in sorted_distances:
            instances.append(k[0][:3])
        # print(instances)
        correct_instances = 0
        precision_total = 0
        for x in range(1, len(instances) + 1):
            if correct_instances == num_of_ids[i]:
                # print(x-1)
                break
            elif instances[x - 1] == query_images_ids[i]:
                correct_instances += 1
                # print(correct_instances, x, correct_instances / x)
                precision_total += correct_instances / x
        total += precision_total / num_of_ids[i]
    return total / len(query_images)


root_dir = "/home/fyp3-2/Desktop/BATCH18/ReID_check/manjula"
query_dir = root_dir + "/query"
gallery_dir = root_dir + "/gallery"
query_images = []
query_images_ids = []
gallery_images = []
gallery_images_ids = []
num_of_ids = []

for root, query_dirs, query_images_names in os.walk(query_dir, topdown=True):
    if len(query_images_names) != 0:
        for i in range(len(query_images_names)):
            if query_images_names[i][-3:] == 'jpg':
                query_images.append(query_images_names[i])
            query_images_ids.append(root[-3:])

for root, gallery_dirs, gallery_images_names in os.walk(gallery_dir, topdown=True):
    if len(gallery_images_names) != 0:
        for i in range(len(gallery_images_names)):
            if gallery_images_names[i][-3:] == 'jpg':
                gallery_images.append(gallery_images_names[i])
            gallery_images_ids.append(root[-3:])
        num_of_ids.append(len(gallery_images_names))

# divider = "###########"
# print(query_images)
# print(divider)
# print(query_images_ids)
# print(divider)
# print(gallery_images)
# print(divider)
# print(gallery_images_ids)
# print(divider)
# print(num_of_ids)

# print(compare(query_images[0], query_images_ids[0], gallery_images[0], gallery_images_ids[0]))
# print(accuracy(query_images, query_images_ids, gallery_images, gallery_images_ids, 10))
# print(mAP(query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids))

top1, top5, top10 = accuracy(query_images, query_images_ids, gallery_images, gallery_images_ids)

MAP = mAP(query_images, query_images_ids, num_of_ids, gallery_images, gallery_images_ids)

print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (top1, top5, top10, MAP))