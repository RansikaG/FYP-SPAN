import argparse
import os

import numpy as np
import torch

import data_preparation
import train
import evaluate

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Re-ID using SPAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', required=True,
                        help='Select training or implementation mode; option: ["train", "implement"]')
    parser.add_argument('--dataset', required=True, help='path to Re-ID dataset')
    parser.add_argument('--part_att_ckpt', required=True, help='path to part attention mask model checkpoint')
    parser.add_argument('--mask_dl_ckpt', required=True, help='path to mask dl model checkpoint')
    parser.add_argument('--mask_dir', required=True, help='path to target part attention mask dir')
    parser.add_argument('--train_csv_path', required=True, help='path to csv file with train data')

    args = parser.parse_args()

    print("### Mask Preparation ###")
#    data_preparation.pipeline_span(Or_image_root=args.dataset,
#                                   mask_dl_ckpt=args.mask_dl_ckpt,
#                                   part_att_ckpt=args.part_att_ckpt,
#                                   target_dir=args.mask_dir)
    print("### ReID Train ###")
    train_data_path = args.dataset + '/train'
    train_mask_path = args.mask_dir + '/attention_masks_gen/train'
#    train.reid_train(csv_path_val='/home/fyp3-2/Desktop/BATCH18/ReID_check/val_data.csv', csv_path_train=args.train_csv_path,
#                     train_data_path=train_data_path, mask_path_train=train_mask_path,
#                     mask_path_val='/home/fyp3-2/Desktop/BATCH18/ReID_check/val_attention_masks_gen/val',
#                     val_data_path='/home/fyp3-2/Desktop/BATCH18/ReID_check/Val_data/val')
    train_mask_path = args.mask_dir + '/attention_masks_gen'
    print('### Evaluate ###')
    evaluate.reid_evaluation(root_dir=args.dataset, mask_dir=train_mask_path)
