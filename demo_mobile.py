# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:12:57 2022

@author: lowes
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 17:04:21 2022

@author: s183983
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:33:51 2022

@author: lowes
"""
import os
import numpy as np
from PIL import Image
import shutil
import requests
import cv2
import imutils
# import glob
path = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/General_Interactive_Segmentation"
import annotator_v2 as ann

medical_images = False
instant_bool = True
negative_skeleton = True


net_name = "amix_150000" #net_list[-2] #"graph_200000"

resize_size = 256 if net_name.split('_')[0].find('256') != -1 else 128


url = "http://192.168.0.113:8080/shot.jpg"

url = "http://10.38.165.93:8080/shot.jpg"

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)
    cv2.imshow("Android_cam", img)
  
    # Press Esc key to exit
    if cv2.waitKey(1) != -1:
        break
  
cv2.destroyAllWindows()
im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image,(1000,800))
ann.annotate(im, net_name,
            instant_seg=instant_bool,
            negative_skeleton=negative_skeleton,
            resize_size=resize_size,
            medical_images = medical_images,
            url = url)

