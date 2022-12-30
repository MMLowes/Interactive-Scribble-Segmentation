# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:33:51 2022

@author: lowes
"""
import os
import numpy as np
from PIL import Image
# import glob
path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation"
os.chdir(path)
import annotator_v2 as ann

medical_images = False

instant_bool = True
negative_skeleton = True


net_list = ["mrbean_190000", "mrbean_ite_200000", "mrbean_unmask_120000",
            "tree_330000", "tree_ite_240000","mix256_40000","amix128_080000",
            "all_110000", "med_060000"]

net_name = "amix_150000" #net_list[-2] #"graph_200000" net_list[-3]


resize_size = 256 if net_name.split('_')[0].find('256') != -1 else 128

if medical_images:
    im_name = os.path.join(path,"CHAOS_Train_Sets/Train_Sets/images","CT02416.png")
else:
    im_name = os.path.join(path,'benchmark/dataset/img/2008_000099.jpg')
# im_name = os.path.join(path,'benchmark/dataset/img/2008_000099.jpg')

im = np.array(Image.open(im_name))
if im.ndim==2:
    im = np.stack((im,im,im),axis=2)
import cv2
im = cv2.resize(im,(1000,800))
ann.annotate(im, net_name,
            instant_seg=instant_bool,
            negative_skeleton=negative_skeleton,
            resize_size=resize_size,
            medical_images = medical_images)

