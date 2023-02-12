# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:33:51 2022

@author: lowes
"""
import os
import numpy as np
from PIL import Image
import annotator_v2 as ann
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p","--path", type=str, help="Path to images", default="images")

args = parser.parse_args()


image_path = args.path

medical_images = False

instant_bool = True
negative_skeleton = True

net_name = "amix_150000" 

if not os.path.isdir("annotations"):
    os.mkdir("annotations")

resize_size = 256 if net_name.split('_')[0].find('256') != -1 else 128


ann.annotate(image_path, net_name,
            instant_seg=instant_bool,
            negative_skeleton=negative_skeleton,
            resize_size=resize_size,
            )
