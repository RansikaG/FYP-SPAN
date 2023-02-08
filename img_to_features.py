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

def weightedFeatures(root_dir, mask_dir):
    gallery_dir = root_dir + "/gallery"
    gallery_mask_dir = mask_dir + "/gallery"
    gallery_images = []
    gallery_images_ids = []
    gallery_images_features = []
    num_of_ids = []
    img_transform = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
    mask_transform = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

    for root, gallery_dirs, gallery_images_names in os.walk(gallery_dir, topdown=True):
        if len(gallery_images_names) != 0:
            for i in range(len(gallery_images_names)):
                if gallery_images_names[i][-3:] == 'jpg':
                    imageGalleryT = img_transform(Image.open(gallery_dir + '/' + root[-3:] + '/' + gallery_images_names[i]).convert('RGB'))
                    imageGallery = torch.unsqueeze(imageGalleryT, 0)
                    frontGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_front.jpg'))
                    frontGallery = torch.unsqueeze(frontGalleryT, 0)
                    rearGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_rear.jpg'))
                    rearGallery = torch.unsqueeze(rearGalleryT, 0)
                    sideGalleryT = mask_transform(Image.open(gallery_mask_dir + '/' + gallery_images_names[i][:-4] + '_side.jpg'))
                    sideGallery = torch.unsqueeze(sideGalleryT, 0)
                    gallery_img_features = model(imageGallery.to(device), frontGallery.to(device), rearGallery.to(device), sideGallery.to(device))
                    gallery_images_features.append(gallery_img_features)
                    gallery_images.append(gallery_images_names[i])
                gallery_images_ids.append(root[-3:])
            num_of_ids.append(len(gallery_images_names))


    x = 0
    for i in num_of_ids:
        while x < len(gallery_images):
            for j in gallery_images_features[x : x + i]:

        x += i


