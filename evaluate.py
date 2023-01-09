import scipy.io
import torch
import numpy as np
#import time
import os
from torch import nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = m_result['mquery_f']
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('accuracy:%f mAP:%f'%(CMC[0],ap/len(query_label)))

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = np.mean(mquery_feature[mquery_index,:], axis=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))


#mine
def calc_euclidean(x1, x1_area_ratio, x2, x2_area_ratio):
    global_feature_size=1024
    part_feature_size=512
    cam = np.array(x1_area_ratio) * np.array(x2_area_ratio)
    normalized_cam = cam / np.sum(cam, axis=1, keepdims=True)
    normalized_cam = torch.from_numpy(normalized_cam).float().to(device)
    distance = (x1 - x2).pow(2)
    global_distance = distance[:, :global_feature_size] * normalized_cam[:, 0:1]
    front_distance = distance[:,
                     global_feature_size: global_feature_size + part_feature_size] * normalized_cam[
                                                                                                    :, 1:2]
    rear_distance = distance[:,
                    global_feature_size + part_feature_size: global_feature_size + 2 * part_feature_size] * normalized_cam[
                                                                                                                                :,
                                                                                                                                2:3]
    side_distance = distance[:, global_feature_size + 2 * part_feature_size:] * normalized_cam[:, 3:]
    weighted_distance = torch.cat((global_distance, front_distance, rear_distance, side_distance), 1).sum(1)
    return weighted_distance


def get_area_ratios(image_name):
    # image_root='./PartAttMask/image_query'
    # image = os.path.join(image_root, image_name)
    front = Image.open(image_name.replace('.jpg', '_front.jpg'))
    front_area = np.sum(np.array(front) / 255)
    rear = Image.open(image_name.replace('.jpg', '_rear.jpg'))
    rear_area = np.sum(np.array(rear) / 255)
    side = Image.open(image_name.replace('.jpg', '_side.jpg'))
    side_area = np.sum(np.array(side) / 255)
    global_area = front_area + rear_area + side_area
    front_area /= global_area
    rear_area /= global_area
    side_area /= global_area
    global_area /= global_area
    # print('global: {} \nfront: {} \nrear:{} \nside: {}'.format(global_area, front_area, rear_area, side_area))
    area_ratios = np.array([global_area, front_area, rear_area, side_area])
    return area_ratios

def compare(query_image, gallery_image):
    query_img_features = model(query_image)  #model eken ena feature vector eka methnarta one
    gallery_img_features = model(gallery_image) #model eken ena feature vector eka methnarta one
    query_area_ratios = calc_euclidean(query_image)
    gallery_area_ratios = calc_euclidean(gallery_image)

    weighted_distance = calc_euclidean(query_img_features, query_area_ratios, gallery_img_features, gallery_area_ratios)
    
    return weighted_distance

def accuracy(query_images, gallery_images):
    top = 1
    final_accuracy = 0
    for i in len(query_images):
        distances = {}
        for j in len(gallery_images):
            weighted_distance = compare(query_images[i], gallery_images[j])
            distances[gallery_images[j]] = weighted_distance
        sorted_distances = sorted(distances.items(), key=lambda x:x[1])
        correct_instances = sorted_distances.keys()[:top].count(query_images[i])  #methana identity eka danna one
        final_accuracy += correct_instances
    return final_accuracy/(top*len(query_images))