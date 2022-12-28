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

img_root = "./test_images/image_test"
mask_root = "./PartAttMask/image_test"



class TripletLossWithCPDM(nn.Module):
    def __init__(self, margin=1.0, global_feature_size=1024, part_feature_size=512):
        super(TripletLossWithCPDM, self).__init__()
        self.margin = margin
        self.global_feature_size = global_feature_size
        self.part_feature_size = part_feature_size

    def calc_euclidean(self, x1, x1_area_ratio, x2, x2_area_ratio):
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
        distance_positive = self.calc_euclidean(anchor, anchor_area_ratio, positive, positive_area_ratio)
        distance_negative = self.calc_euclidean(anchor, anchor_area_ratio, negative, negative_area_ratio)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


if __name__ == '__main__':

    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    csv_path = 'test_images/identities_train/train_data.csv'
    train_data_path = './test_images/identities_train'
    mask_path = './PartAttMask/image_train'

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float}
    dataframe = pd.read_csv(csv_path, dtype=types_dict)
    dataframe['area_ratios'] = dataframe[['global', 'front', 'rear', 'side']].values.tolist()

    dataset = ImageMasksTriplet(df=dataframe, image_path=train_data_path, mask_path=mask_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, prefetch_factor=2,
                            persistent_workers=True)

    # classifier = model.BoatIDClassifier(num_of_classes=5)
    model = model.Second_Stage_Resnet50_Features()

    if torch.cuda.is_available():
        model.cuda()
        # classifier.cuda()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    activation = {}
    def get_activation(name):
        def hook(model, image, output):
            activation[name] = output.detach()
        return hook

    model.stage2_features_global.register_forward_hook(get_activation('global'))
    model.stage2_features_front.register_forward_hook(get_activation('front'))
    model.stage2_features_rear.register_forward_hook(get_activation('rear'))
    model.stage2_features_side.register_forward_hook(get_activation('side'))


    model.eval()
    pbar = tqdm(total=len(dataloader))
    for batch_idx, data in enumerate(dataloader):
        anchor_img, anchor_image_masks, _, _, _, _, _, _, _, _ = data

        anchor_img_features = model(anchor_img.to(device), anchor_image_masks[0].to(device),
                                    anchor_image_masks[1].to(device), anchor_image_masks[2].to(device))

        print(activation['global'].shape)
        print(activation['front'].shape)
        print(activation['rear'].shape)
        print(activation['side'].shape)

        pbar.set_postfix({'Batch no': ' {}'.format(batch_idx + 1)})
        pbar.update(1)
    pbar.close()
