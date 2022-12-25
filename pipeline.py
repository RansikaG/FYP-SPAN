import torch.optim as optim
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from ImageMasksDataset import ImageMasksTriplet
from resnet import CNN1, CNN2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser(description='Train Semantics-guided Part Attention Network (SPAN) pipeline', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--mode', required=True, help='Select training or implementation mode; option: ["train", "implement"]')

img_root = "./test_images/image_test"
mask_root = "./PartAttMask/image_test"


class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size=1024, part_feature_size=512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_euclidean(self, x1, x1_area_ratio, x2, x2_area_ratio):
        cam = x1_area_ratio * x2_area_ratio
        # print(cam)
        normalized_cam = cam / np.sum(cam)
        distance = (x1 - x2).pow(2).sum(1)
        weighted_distance = np.concatenate((distance[:self.global_feature_size] * normalized_cam[0],
                                            distance[
                                            self.global_feature_size: self.global_feature_size + self.part_feature_size] *
                                            normalized_cam[1],
                                            distance[
                                            self.global_feature_size + self.part_feature_size: self.global_feature_size + 2 * self.part_feature_size] *
                                            normalized_cam[2],
                                            distance[self.global_feature_size + 2 * self.part_feature_size:] *
                                            normalized_cam[3]))
        return weighted_distance

    def forward(self, anchor, anchor_area_ratio, positive, positive_area_ratio, negative, negative_area_ratio):
        distance_positive = self.calc_euclidean(anchor, anchor_area_ratio, positive, positive_area_ratio)
        distance_negative = self.calc_euclidean(anchor, anchor_area_ratio, negative, negative_area_ratio)
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

    csv_path = 'test_images/identities_train/train_data.csv'
    train_data_path = './test_images/identities_train'
    mask_path = './PartAttMask/image_train'

    dataframe = pd.read_csv(csv_path)

    dataset = ImageMasksTriplet(df=dataframe, image_path=train_data_path, mask_path=mask_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2,
                            persistent_workers=True)

    model = model.Second_Stage_Extractor()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epoch = 2
    for ep in range(epoch):
        model.train()
        print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
        pbar = tqdm(total=len(dataloader))
        for batch_idx, data in enumerate(dataloader):
            anchor_img, anchor_image_masks, anchor_area_ratios, positive_img, \
            positive_img_masks, positive_area_ratios, negative_img, negative_img_masks,\
            negative_area_ratios, anchor_label = data.to(device)

            anchor_img_features = model(anchor_img, anchor_image_masks)
            positive_img_features = model(positive_img, positive_img_masks)
            negative_img_features = model(negative_img, negative_img_masks)

            criterion = TripletLossWithCPDM()
            loss = criterion(anchor_img_features, anchor_area_ratios, positive_img_features,
                            positive_area_ratios, negative_img_features, negative_area_ratios)
            # loss = loss_mask + loss_area + loss_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'Triplet_loss': ' {0:1.3f}'.format(loss / (batch_idx + 1))})
            pbar.update(1)
        pbar.close()



            # print(f'image size: {image.size()}\n front: {front_mask.size()}\n rear: {rear_mask.size()}\n'
            #       f' side: {side_mask.size()}\n CNN1: {stage_1_CNN.size()}\n global_features: {global_features.size()}')
    # print(summary(CNN1, (3, 192, 192)))
    # print(summary(CNN2, (1024, 24, 24)))
    # mask_generator = model.PartAtt_Generator().to(device)
    # mask_generator.eval()
    # mask_generator.load_state_dict(torch.load('./PartAttMask_ckpt/10.ckpt'))
