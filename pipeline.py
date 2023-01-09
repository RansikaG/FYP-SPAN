import os

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import model
from ImageMasksDataset import ImageAndMasksFeatures, ImageFeatures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Train = True
feature_extraction = False

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
    feature_tensor_save_path = './test_images/identities_train/features'

    types_dict = {'filename': str, 'id': str, 'global': float, 'front': float, 'rear': float, 'side': float}
    dataframe = pd.read_csv(csv_path, dtype=types_dict)
    dataframe['area_ratios'] = dataframe[['global', 'front', 'rear', 'side']].values.tolist()

    if feature_extraction:
        img_dataset = ImageAndMasksFeatures(df=dataframe, image_path=train_data_path, mask_path=mask_path)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False)

        # classifier = model.BoatIDClassifier(num_of_classes=5)
        resnetExtractor = model.Second_Stage_Resnet50_Features()

        if torch.cuda.is_available():
            resnetExtractor.cuda()

        activation = {}


        def get_activation(name):
            def hook(model, image, output):
                activation[name] = output.detach()

            return hook


        resnetExtractor.stage2_features_global.register_forward_hook(get_activation('global'))
        resnetExtractor.stage2_features_front.register_forward_hook(get_activation('front'))
        resnetExtractor.stage2_features_rear.register_forward_hook(get_activation('rear'))
        resnetExtractor.stage2_features_side.register_forward_hook(get_activation('side'))

        resnetExtractor.eval()
        pbar = tqdm(total=len(img_dataloader))

        print('#### preparing image features')
        if not os.path.isdir(feature_tensor_save_path):
            os.mkdir(feature_tensor_save_path)
        for batch_idx, data in enumerate(img_dataloader):
            img, image_masks, img_name = data

            img_features = resnetExtractor(img.to(device), image_masks[0].to(device),
                                           image_masks[1].to(device), image_masks[2].to(device))

            global_features = activation['global']
            front_features = activation['front']
            rear_features = activation['rear']
            side_features = activation['side']

            # torch.save(global_features, feature_tensor_save_path + '/' + img_name[0].replace('.jpg', '_global.pt'))
            # torch.save(front_features, feature_tensor_save_path + '/' + img_name[0].replace('.jpg', '_front.pt'))
            # torch.save(rear_features, feature_tensor_save_path + '/' + img_name[0].replace('.jpg', '_rear.pt'))
            # torch.save(side_features, feature_tensor_save_path + '/' + img_name[0].replace('.jpg', '_side.pt'))
            torch.save(torch.squeeze(img_features), feature_tensor_save_path + '/' + img_name[0].replace('.jpg', '.pt'))
            pbar.set_postfix({'Img no': ' {}'.format(batch_idx + 1)})
            pbar.update(1)
        pbar.close()
        del resnetExtractor, img_dataset, img_dataloader
    print('#### Training #####')
    if Train:
        feature_dataset = ImageFeatures(df=dataframe, feature_path=feature_tensor_save_path, device=device)
        feature_dataloader = DataLoader(feature_dataset, batch_size=2, shuffle=False)

        classifier = model.BoatIDClassifier(num_of_classes=5)
        fc_model = model.FC_Features()

        if torch.cuda.is_available():
            fc_model.cuda()
            classifier.cuda()
        optimizer = optim.Adam(fc_model.parameters(), lr=0.0001)

        epoch = 2

        for ep in range(epoch):
            fc_model.train()
            classifier.train()
            print('\nStarting epoch %d / %d :' % (ep + 1, epoch))
            pbar = tqdm(total=len(feature_dataloader))
            for batch_idx, data in enumerate(feature_dataloader):
                anchor_raw_features, anchor_area_ratios, positive_raw_features, positive_area_ratios, \
                negative_raw_features, negative_area_ratios, target = data

                anchor_img_features = fc_model(anchor_raw_features)
                positive_img_features = fc_model(positive_raw_features)
                negative_img_features = fc_model(negative_raw_features)

                prediction = classifier(anchor_img_features)
                criterion1 = nn.CrossEntropyLoss()
                criterion2 = TripletLossWithCPDM()

                cross_entropy_loss = criterion1(prediction, target.to(device))
                triplet_loss = criterion2(anchor_img_features, anchor_area_ratios, positive_img_features,
                                          positive_area_ratios, negative_img_features, negative_area_ratios)

                lambda_ID = 1
                lambda_triplet = 1
                loss = lambda_ID * cross_entropy_loss + lambda_triplet * triplet_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'Triplet_loss': ' {0:1.3f}'.format(loss / (batch_idx + 1))})
                pbar.update(1)
            pbar.close()
