import argparse
import os

import numpy as np
import torch

import data_preparation

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Re-ID using SPAN',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', required=True,
                        help='Select training or implementation mode; option: ["train", "implement"]')
    parser.add_argument('--dataset', required=True, help='path to Re-ID dataset')
    parser.add_argument('--part_att_ckpt', required=True, help='path to part attention mask model checkpoint')
    parser.add_argument('--mask_dir', required=True, help='path to target part attention mask dir')

    args = parser.parse_args()

    data_preparation.pipeline_span(Or_image_root=args.dataset,
                                   part_att_ckpt=args.part_att_ckpt,
                                   target_dir=args.mask_dir)
