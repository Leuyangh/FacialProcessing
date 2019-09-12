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
savingpath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\aligned3DLfwFeatureSmallTest\\"

identities = glob(datasetPath+"*")
count = 0
inputPicTotal = 0
outputPicTotal = 0
#leyeminx, leyeminy, reyeminx, reyeminy, noseminx, noseminy, mouthminx, mouthminy = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
#leyemaxx, leyemaxy, reyemaxx, reyemaxy, nosemaxx, nosemaxy, mouthmaxx, mouthmaxy = [-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000]
badpics = 0

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
            if i == 0:

                reyeminy = min(y, reyeminy)
                reyemaxy = max(y, reyemaxy)
                reyeminx = min(x, reyeminx)
                reyemaxx = max(x, reyemaxx)

                crop_img = image[y-10:y+10, x-10:x+10].copy()
                if x-10 < 0 | y-10 < 0:
                    print ("bad image: " + picID + " " + areanames[i])
                    badpics += 1
                else:
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, crop_img)
                    outputPicTotal += 1
            if i == 1:
                leyeminy = min(y, leyeminy)
                leyemaxy = max(y, leyemaxy)
                leyeminx = min(x, leyeminx)
                leyemaxx = max(x, leyemaxx)
                crop_img = image[y-10:y+10, x-10:x+10].copy()
                if x+10 > 64 | y-10 < 0:
                    print ("bad image: " + picID + " " + areanames[i])
                    badpics += 1
                else:
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, crop_img)
                    outputPicTotal += 1
            if i == 2:
                #print ("Right eyebrow at: " + str(x) + "," + str(y))
                if y < 0:
                    x+= 5
                    pad = -y
                    crop_img = image[0:y+20, x-10:x+10].copy()
                    padded_img = cv2.copyMakeBorder(crop_img, top=pad, bottom=0, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, padded_img)
                    outputPicTotal += 1
                else:
                    x += 5
                    y += 10
                    crop_img = image[y-10:y+10, x-10:x+10].copy()
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, crop_img)
                    outputPicTotal += 1
            if i == 3:
                #print ("Left eyebrow at: " + str(x) + "," + str(y))
                if y < 0:
                    x-= 5
                    pad = -y
                    crop_img = image[0:y+20, x-10:x+10].copy()
                    padded_img = cv2.copyMakeBorder(crop_img, top=pad, bottom=0, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, padded_img)
                    outputPicTotal += 1
                else:
                    x -= 5
                    y += 10
                    crop_img = image[y-10:y+10, x-10:x+10].copy()
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, crop_img)
                    outputPicTotal += 1
            if i == 4:
                noseminy = min(y, noseminy)
                nosemaxy = max(y, nosemaxy)
                noseminx = min(x, noseminx)
                nosemaxx = max(x, nosemaxx)
                crop_img = image[y-10:y+10, x-10:x+10].copy()
                dest_path = savingpath + areanames[i]
                os.makedirs(dest_path, exist_ok=True)
                finalPath = dest_path + picID + ".jpg"
                cv2.imwrite(finalPath, crop_img)
                outputPicTotal += 1
            if i == 5:
                mouthminy = min(y, mouthminy)
                mouthmaxy = max(y, mouthmaxy)
                mouthminx = min(x, mouthminx)
                mouthmaxx = max(x, mouthmaxx)
                y -= 7
                crop_img = image[y-10:y+10, x-10:x+10].copy()
                if y+10 > 64:
                    print ("bad image: " + picID + " " + areanames[i])
                    badpics += 1
                else:
                    dest_path = savingpath + areanames[i]
                    os.makedirs(dest_path, exist_ok=True)
                    finalPath = dest_path + picID + ".jpg"
                    cv2.imwrite(finalPath, crop_img)
                    outputPicTotal += 1
                
"""
print("Left eye min x: " + str(leyeminx))
print("Left eye max x: " + str(leyemaxx))
print("Left eye min y: " + str(leyeminy))
print("Left eye max y: " + str(leyemaxy))
print("Right eye min x: " + str(reyeminx))
print("Right eye max x: " + str(reyemaxx))
print("Right eye min y: " + str(reyeminy))
print("Right eye max y: " + str(reyemaxy))
print("Nose min x: " + str(noseminx))
print("Nose max x: " + str(nosemaxx))
print("Nose min y: " + str(noseminy))
print("Nose max y: " + str(nosemaxy))
print("Mouth min x: " + str(mouthminx))
print("Mouth max x: " + str(mouthmaxx))
print("Mouth min y: " + str(mouthminy))
print("Mouth max y: " + str(mouthmaxy))
"""

print("Created " + str(outputPicTotal) + " pictures from " + str(inputPicTotal) + " aligned images with " + str(badpics) + " bad images")
