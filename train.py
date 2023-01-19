import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from ImageMasksDataset import ImageMasksTriplet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description='Train Semantics-guided Part Attention Network (SPAN) pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--mode', required=True, help='Select training or implementation mode; option: ["train", "implement"]')

img_root = "/home/fyp3-2/Desktop/BATCH18/ReID_check/train"
mask_root = "/home/fyp3-2/Desktop/BATCH18/ReID_check/masks_train"


class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size=1024, part_feature_size=512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_distance_vector(self, x1, x1_area_ratio, x2, x2_area_ratio):
        cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
        normalized_cam = cam / np.sum(cam, axis=1, keepdims=True)
        normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
        distance = (x1 - x2).pow(2)
        global_distance = distance[:, :self.global_feature_size] * normalized_cam[:, 0:1]
        front_distance = distance[:,
                         self.global_feature_size: self.global_feature_size + self.part_feature_size] * normalized_cam[
                                                                                                        :, 1:2]
        rear_distance = distance[:,
                        self.global_feature_size + self.part_feature_size: self.global_feature_size + 2 * self.part_feature_size] * normalized_cam[
                                                                                                                                    :,
                                                                                                                                    2:3]
        side_distance = distance[:, self.global_feature_size + 2 * self.part_feature_size:] * normalized_cam[:, 3:]

        weighted_distance = torch.cat((global_distance, front_distance, rear_distance, side_distance), 1).sum(1)
        return weighted_distance

    def forward(self, anchor, anchor_area_ratio, positive, positive_area_ratio, negative, negative_area_ratio):
        distance_positive = self.calc_distance_vector(anchor, anchor_area_ratio, positive, positive_area_ratio)
        distance_negative = self.calc_distance_vector(anchor, anchor_area_ratio, negative, negative_area_ratio)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


if __name__ == '__main__':

    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    csv_path = "/home/fyp3-2/Desktop/BATCH18/ReID_check/train_data.csv"
    train_data_path = "/home/fyp3-2/Desktop/BATCH18/ReID_check/train"
    mask_path = "/home/fyp3-2/Desktop/BATCH18/ReID_check/masks_train"

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float}
    dataframe = pd.read_csv(csv_path, dtype=types_dict)
    dataframe['area_ratios'] = dataframe[['global', 'front', 'rear', 'side']].values.tolist()

    dataset = ImageMasksTriplet(df=dataframe, image_path=train_data_path, mask_path=mask_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2,
    #                         # persistent_workers=True)

    classifier = model.BoatIDClassifier(num_of_classes=474)
    model = model.Second_Stage_Extractor()

    if torch.cuda.is_available():
        model.cuda()
        classifier.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    epoch = 1

    triplet_loss_bucket=[]
    CE_loss_bucket = []
    total_loss_bucket = []
    for ep in range(epoch):
        model.train()
        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader))
        for batch_idx, data in enumerate(dataloader):
            anchor_img, anchor_image_masks, anchor_area_ratios, positive_img, \
            positive_img_masks, positive_area_ratios, negative_img, negative_img_masks, \
            negative_area_ratios, target = data

            anchor_img_features = model(anchor_img.to(device), anchor_image_masks[0].to(device),
                                        anchor_image_masks[1].to(device), anchor_image_masks[2].to(device))
            positive_img_features = model(positive_img.to(device), positive_img_masks[0].to(device),
                                          positive_img_masks[1].to(device), positive_img_masks[2].to(device))
            negative_img_features = model(negative_img.to(device), negative_img_masks[0].to(device),
                                          negative_img_masks[1].to(device), negative_img_masks[2].to(device))

            prediction = classifier(anchor_img_features)
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = TripletLossWithCPDM()

            cross_entropy_loss = criterion1(prediction, target.to(device))
            triplet_loss = criterion2(anchor_img_features, anchor_area_ratios, positive_img_features,
                                      positive_area_ratios, negative_img_features, negative_area_ratios)

            lambda_ID = 1
            lambda_triplet = 1
            loss = lambda_ID * cross_entropy_loss + lambda_triplet * triplet_loss

            triplet_loss_bucket.append(triplet_loss)
            CE_loss_bucket.append(cross_entropy_loss)
            total_loss_bucket.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Triplet_loss': ' {0:1.6f}'.format(triplet_loss/len(dataloader)), 'ID_loss': ' {0:1.6f}'.format(cross_entropy_loss/len(dataloader))})
            pbar.update(1)
        pbar.close()
    torch.save(model, "/home/fyp3-2/Desktop/BATCH18/ReID_check/temp.pth")