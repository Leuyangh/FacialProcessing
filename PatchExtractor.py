# -*- coding: utf-8 -*-
"""

@author: Eric
"""

import argparse
from glob import glob
import cv2
from imutils.face_utils.helpers import shape_to_np
import imutils
import dlib
import numpy as np
import os

def maxbound(value):
    if value > 64:
        return 64
    else:
        return value
def minbound(value):
    if value < 0:
        return 0
    else:
        return value

predictor = dlib.shape_predictor("C:\\Users\\Eric\\Documents\\Face_recognition_project\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat")

"""PATCH EXTRACTION"""

datasetPath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\aligned3DLfw\\"
savingpath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\aligned3DLfwFeatures\\"

identities = glob(datasetPath+"*")
count = 0
inputPicTotal = 0
outputPicTotal = 0
for identity in identities:
    count += 1
    identityStr = os.path.basename(identity).split('\\')[-1]
    pics = glob(identity+"\\*")
    print("Creating patches for " + identityStr + "'s " + str(len(pics)) + " picture(s)")
    inputPicTotal += len(pics)
    for pic in pics:
        picID = os.path.basename(pic).split('\\')[-1]
        picID = picID.split('.')[0]
        image = cv2.imread(pic)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(0, 0, 64, 64)#sketchy manually defined bounding box. saves time over using dnnfacedetector again because images are assured to be 64x64
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        #landmarks are 42, 47, 20, 25, 34, and 67 for right eye, left eye, right eyebrow, left eyebrow, nose, and mouth. 
        #decrement by 1 to account for starting at 0 not 1 -> 41, 46, 19, 24, 33, 66
        centerpts = [41, 46, 19, 24, 33, 66]
        #putting numbers before names to make sure theyre in the right order in the folders
        areanames = ["right_eyes\\", "left_eyes\\", "right_eyebrows\\", "left_eyebrows\\", "noses\\", "mouths\\"]
        for i,point in enumerate(centerpts):
            x, y = shape[point]
            ymin = minbound(y-15)
            ymax = maxbound(y+15)
            xmin = minbound(x-15)
            xmax = maxbound(x+15)
            crop_img = image[ymin:ymax, xmin:xmax].copy()
            pointName = areanames[i]
            dest_path = savingpath + pointName
            os.makedirs(dest_path, exist_ok=True)
            finalPath = dest_path + picID + ".jpg"
            cv2.imwrite(finalPath, crop_img)
            outputPicTotal += 1
        
print("Created " + str(outputPicTotal) + " pictures from " + str(inputPicTotal) + " aligned images")

"""
HE Code
"""

datasetPath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\aligned3DLfwFeatures\\"
savingpath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\HE3DLfwFeatures\\"
identities = glob(datasetPath+"*")
count=0

for identity in identities:
    count+=1
    identityStr = os.path.basename(identity).split('\\')[-1]
    pics=glob(identity+"\\*")
    print("creating paths for identity " +identityStr)
    dest_path= savingpath + identityStr + "\\"
    os.makedirs(dest_path, exist_ok=True)
    for pic in pics:
        picId=os.path.basename(pic).split('\\')[-1]
        img = cv2.imread(pic)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img)
        cv2.imwrite(dest_path + picId, equ)
