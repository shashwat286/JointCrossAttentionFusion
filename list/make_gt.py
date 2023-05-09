import numpy as np
import os
import cv2
from replace import replace,delete

clip_len = 16

# the dir of testing images
video_root = 'D:/FYP 2023 - Group 43/Dataset/videos'  # the path of test videos
feature_list = 'list/rgb_test_new.list'
# the ground truth txt
gt_txt = 'xx/annotations.txt'  # the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0
for idx in range(len(lists)):
    name = lists[idx].strip('\n').split('/')[-1]
    name = delete(name,"RGBTest")
    if '__0.npy' not in name:
        continue
    name = name[:-7]
    name = name[1:]
    vname = name+'.mp4'
    vname = replace(vname)
    path = os.path.join(video_root, vname)
    cap = cv2.VideoCapture(os.path.join(video_root, vname))
    lens = int(cap.get(7))  # get frame nums

    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if '_label_A' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                        for j in range(2,len(gt_content),2) if j==i+1]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        gt_vec[frag[0]:frag[1]]=1.0
                break
    mod = (lens-1) % clip_len  # minusing 1 is to align flow  rgb: minusing 1 when extracting features
    gt_vec = gt_vec[:-1]
    if mod:
        gt_vec = gt_vec[:-mod]
    interval = np.linspace(0, len(gt_vec)/16, 50, dtype=np.uint16)
    gt_vec_new = np.array([])
    for i in interval:
        np.append(gt_vec_new,gt_vec[i*16:(i*16)+16])
    gt.extend(gt_vec_new)
    if sum(gt_vec_new)/len(gt_vec_new):
        tlens += len(gt_vec_new)
        vlens += sum(gt_vec_new)
    count = idx

np.save('list/gt_new_.npy', gt)
