import argparse
import glob
import os
import torch
from resnet import resnet34
from PIL import Image, ImageOps
from CPDM import CPDM
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
        image = Image.open(self.filenames[index]).resize((192,192))
        image = ImageOps.grayscale(image)
        front_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_front.jpg'))
        rear_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_rear.jpg'))
        side_mask = Image.open(
            self.filenames[index].replace(self.image_root, self.mask_root).replace('.jpg', '_side.jpg'))
        # mask = np.array(mask.resize((60, 60)))
        # mask = torch.from_numpy(mask / 255).float()
        convert_tensor = transforms.ToTensor()

        return convert_tensor(image), convert_tensor(front_mask), convert_tensor(rear_mask),\
            convert_tensor(side_mask)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    model = resnet34().to(device)
    model.eval()

    # transform = T.Compose([T.Resize([192, 192]),
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = ImageAndMasks(image_root=img_root, mask_root=mask_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        for _, data in enumerate(dataloader):
            image, front_mask, rear_mask, side_mask = data
            print("image size: {}\n front: {}\n rear: {}\n side: {}\n".format(image.size(), front_mask.size(),
                                                                              rear_mask.size(), side_mask.size()))
            # masks = model(data.to(device))
            # masks = masks.detach().cpu().numpy()
            # for idx, mask in enumerate(masks):
            #     cv2.imwrite(filenames[idx].replace(image_root, mask_root), mask * 255)
            pbar.update(1)
        pbar.close()
