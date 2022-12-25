import glob
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

import model
from CPDM import CPDM
from resnet import CNN1, CNN2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description='Train Semantics-guided Part Attention Network (SPAN) pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--mode', required=True, help='Select training or implementation mode; option: ["train", "implement"]')

areas = CPDM()
areas.get_area_ratios(image_name="Acura_ILX_2019_25_17_200_24_4_70_55_182_24_FWD_5_4_4dr_rMu.jpg")
areas.cooccurence_attention(image_1="Acura_ILX_2019_25_17_200_24_4_70_55_182_24_FWD_5_4_4dr_rMu.jpg",
                            image_2="Acura_MDX_2019_44_18_290_35_6_77_67_196_20_FWD_7_4_SUV_ueu.jpg")

img_root = "./test_images/image_test"
mask_root = "./PartAttMask/image_test"


class ImageAndMasks(Dataset):
    def __init__(self, image_root, mask_root=None):
        self.image_root = image_root
        self.mask_root = mask_root
        self.filenames = glob.glob(os.path.join(image_root, '*.jpg'))
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).resize((192, 192))
        # image = ImageOps.grayscale(image)
        convert_tensor = transforms.Compose([transforms.Resize([192, 192]), transforms.ToTensor()])
        transform_mask = transforms.Compose([transforms.Resize([24, 24]), transforms.ToTensor()])

        front_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_front.jpg'))
        rear_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_rear.jpg'))
        side_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_side.jpg'))
        # mask = np.array(mask.resize((60, 60)))
        # mask = torch.from_numpy(mask / 255).float()

        return convert_tensor(image), transform_mask(front_mask), transform_mask(rear_mask), transform_mask(side_mask)

    def __len__(self):
        return self.len


class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size = 1024, part_feature_size = 512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_euclidean(self, x1, x1_area_ratio, x2, x2_area_ratio):
        cam = x1_area_ratio * x2_area_ratio
        # print(cam)
        normalized_cam = cam / np.sum(cam)
        distance = (x1 - x2).pow(2).sum(1)
        weighted_distance = np.concatenate((distance[:self.global_feature_size]*normalized_cam[0],
             distance[self.global_feature_size: self.global_feature_size+self.part_feature_size]*normalized_cam[1],
             distance[self.global_feature_size+self.part_feature_size: self.global_feature_size+2*self.part_feature_size]*normalized_cam[2],
             distance[self.global_feature_size+2*self.part_feature_size:]*normalized_cam[3]))
        return weighted_distance

    def forward(self, anchor, anchor_area_ratio, positive, positive_area_ratio, negative, negative_area_ratio):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


if __name__ == '__main__':
    CNN1 = CNN1().to(device)
    CNN1.eval()

    CNN2 = CNN2().to(device)
    CNN2.eval()

    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = ImageAndMasks(image_root=img_root, mask_root=mask_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=2, persistent_workers=True)

    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        for _, data in enumerate(dataloader):
            image, front_mask, rear_mask, side_mask = data.to(device)

            stage_1_CNN = CNN1(image.to(device))

            front_image = torch.mul(stage_1_CNN, front_mask.to(device))
            rear_image = torch.mul(stage_1_CNN, rear_mask.to(device))
            side_image = torch.mul(stage_1_CNN, side_mask.to(device))

            global_features = CNN2(stage_1_CNN)
            front_features = CNN2(front_image)
            rear_features = CNN2(rear_image)
            side_features = CNN2(side_image)

            print(f'image size: {image.size()}\n front: {front_mask.size()}\n rear: {rear_mask.size()}\n'
                  f' side: {side_mask.size()}\n CNN1: {stage_1_CNN.size()}\n global_features: {global_features.size()}')

            pbar.update(1)
        pbar.close()
    # print(summary(CNN1, (3, 192, 192)))
    # print(summary(CNN2, (1024, 24, 24)))
    # mask_generator = model.PartAtt_Generator().to(device)
    # mask_generator.eval()
    # mask_generator.load_state_dict(torch.load('./PartAttMask_ckpt/10.ckpt'))
