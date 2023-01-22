import torch
from torch.backends import cudnn
import numpy as np
import os, argparse, random

import BGRemove_GrabCut
import BGRemove_DL
import PartAttGen
from visualize import visualize
import model


if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Train Semantics-guided Part Attention Network (SPAN)', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', required=True, help='Select training or implementation mode; option: ["train", "implement"]')
    parser.add_argument('--image_root', required=True, help='path to VeRi-776 dataset')
    parser.add_argument('--mask_grabcut_root', default='./BGRemove_GrabCut', help='path to foreground mask generated by GrabCut')
    parser.add_argument('--mask_dl_root', default='./BGRemove_DL', help='path to foreground mask generated by deep learning network')
    parser.add_argument('--mask_dl_ckpt', default='./BGRemove_DL_ckpt', help='path to store foreground mask generator checkpoint')
    parser.add_argument('--dataset_csv', default='./dataset.csv', help='Dataset csv file used for training part attention mask generator')
    parser.add_argument('--part_att_root', default='./PartAttMask', help='path to generated part attention mask')
    parser.add_argument('--part_att_ckpt', default='./PartAttMask_ckpt', help='path to store part attention mask generator checkpoint')
    args = parser.parse_args()

    if args.mode == 'train':
        print("\n### STEP 1 : Generate foreground mask by GrabCut ###")
        # BGRemove_GrabCut.implement(image_root=args.image_root,
        #                            mask_root=args.mask_grabcut_root)
        #visualize(image_root=args.image_root, foreground_grabcut_root=args.mask_grabcut_root)

        print("\n### STEP 2 : Train foreground mask generator ###")
        # BGRemove_DL.train(image_root=args.image_root,
        #                   mask_root=args.mask_grabcut_root,
        #                   model=model.Foreground_Generator().to(device),
        #                   device=device,
        #                   checkpoint_path=args.mask_dl_ckpt,
        #                   epoch=5)

        print("\n### STEP 3 : Generate foreground mask by deep generator ###")
        checkpoint = os.path.join(args.mask_dl_ckpt, '5.ckpt')
        BGRemove_DL.implement(image_root=args.image_root,
                              mask_root=args.mask_dl_root,
                              model=model.Foreground_Generator().to(device),
                              device=device,
                              checkpoint=checkpoint)
        #visualize(image_root=args.image_root, foreground_grabcut_root=args.mask_grabcut_root, foreground_dl_root=args.mask_dl_root)

        print("\n### STEP 4 : Train part attention mask generator ###")
        # PartAttGen.train(image_root=args.image_root,
        #                  mask_root=args.mask_dl_root,
        #                  csv_file=args.dataset_csv,
        #                  model=model.PartAtt_Generator().to(device),
        #                  device=device,
        #                  checkpoint_path=args.part_att_ckpt,
        #                  epoch=10)

        print("\n### STEP 5 : Generate part attention mask ###")
        checkpoint = os.path.join(args.part_att_ckpt, '10.ckpt')
        PartAttGen.implement(image_root=args.image_root,
                             mask_root=args.part_att_root,
                             model=model.PartAtt_Generator().to(device),
                             device=device,
                             checkpoint=checkpoint)

        print("\n### STEP 6 : Visualization ###")
        visualize(image_root=args.image_root,
                  foreground_grabcut_root=args.mask_grabcut_root,
                  foreground_dl_root=args.mask_dl_root,
                  partmask_root=args.part_att_root)

    elif args.mode == 'implement':
        print("\n### Generate part attention mask ###")
        checkpoint = os.path.join(args.part_att_ckpt, '10.ckpt')
        PartAttGen.implement(image_root=args.image_root,
                             mask_root=args.part_att_root,
                             model=model.PartAtt_Generator().to(device),
                             device=device,
                             checkpoint=checkpoint)
        
        print("\n### Visualization ###")
        # visualize(image_root=args.image_root,
        #           partmask_root=args.part_att_root)
    
    else:
        print("Unsupported mode selection\nOption: ['train', 'implement']")


