# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:56:38 2022

@author: lowes
"""

import os
import numpy as np
from PIL import Image
import glob
path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation"
os.chdir(path)
import annotator_skelet as ann

filelist = glob.glob(os.path.join(path,"VOC2012/JPEGImages","*jpg"))
                     
label_path = os.path.join(path, "VOC2012/SegmentationObject")
                     
save_path = os.path.join(path, "annotated_skeletons/skeletons")

ann.annotate(filelist, label_path, save_path)

