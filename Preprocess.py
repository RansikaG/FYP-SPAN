import os
from CPDM import CPDM
import csv


def create_folder_name(name):
    folder_name = '_'.join(name.split('_')[:-1])
    return folder_name


# def create_csv_with_area_ratios(image_path, mask_path, csv_path, save_path):
#     #creating folders with car name as identity
#     if not os.path.isdir(save_path):
#         os.mkdir(save_path)
#     for root, dirs, files in os.walk(image_path, topdown=True):
#         for name in files:
#             if not name[-3:] == 'jpg':
#                 continue
#             folder_name = create_folder_name(name)
#             src_path = paths[0] + '/' + name
#             src_label_path = paths[1] + '/' + name[:-3] + 'txt'
#             dst_path = save_path + '/' + folder_name
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             save_grayscale(src_path, dst_path + '/' + name[:-3] + 'jpg')

def create_csv_with_area_ratios(image_path, mask_path, csv_path):
    first = True
    with open(csv_path + '/train_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'id', 'global', 'front', 'rear', 'side'])

        for root, dirs, files in os.walk(image_path, topdown=True):
            if first:
                first = False
                continue
            else:
                for image_name in files:
                    cpdm = CPDM(mask_root=mask_path)
                    area_ratios = cpdm.get_area_ratios(image_name=image_name)
                    # area_ratios_csv = ','.join(area_ratios)
                    ID = root[-2:]
                    writer.writerow([image_name, ID, area_ratios[0], area_ratios[1], area_ratios[2], area_ratios[3]])


if __name__ == '__main__':
    create_csv_with_area_ratios(image_path='./test_images/identities_train', mask_path='./PartAttMask/image_train',
                                csv_path='./test_images/identities_train')
