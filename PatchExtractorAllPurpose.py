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


predictor = dlib.shape_predictor("C:\\Users\\Eric\\Documents\\Face_recognition_project\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat")

"""PATCH EXTRACTION"""

datasetPath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\Frontalized_LFW\\Frontalized_LFW\\"
savingpath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\Frontalized_LFWPatchesLarge\\"
helperPath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\Frontalized_LFWHelp\\"

identities = glob(datasetPath+"*")
count = 0
inputPicTotal = 0
outputPicTotal = 0
badpics = 0
patchsize = 25 #in both direction
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
        h, w, c = image.shape
        rect = dlib.rectangle(0, 0, h, w)#sketchy manually defined bounding box. saves time over using dnnfacedetector again because images are assured to be 64x64
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        #landmarks are 42, 47, 20, 25, 34, and 67 for right eye, left eye, right eyebrow, left eyebrow, nose, and mouth. 
        #decrement by 1 to account for starting at 0 not 1 -> 41, 46, 19, 24, 33, 66
        centerpts = [41, 46, 19, 24, 33, 66]
        #putting numbers before names to make sure theyre in the right order in the folders
        areanames = ["right_eyes\\", "left_eyes\\", "right_eyebrows\\", "left_eyebrows\\", "noses\\", "mouths\\"]
        for i,point in enumerate(centerpts):
            x, y = shape[point]
            ymin = y - patchsize
            ymax = y + patchsize
            xmin = x - patchsize
            xmax = x + patchsize
            padsides = [0,0,0,0]
            xpad, ypad = [0,0]
            if xmin < 0:
                xpad = -xmin
                padsides[0] = 1
            if xmax > w:
                xpad = xmax - w
                padsides[1] = 1
            if ymin < 0:
                ypad = -ymin
                padsides[2] = 1
            if ymax > h:
                ypad = ymax - h
                padsides[3] = 1
            else:
                pass
            if (ypad == 0 and xpad == 0):
                crop_img = image[ymin:ymax, xmin:xmax].copy()
                dest_path = savingpath + areanames[i]
                os.makedirs(dest_path, exist_ok=True)
                finalPath = dest_path + picID + ".jpg"
                cv2.imwrite(finalPath, crop_img)
                outputPicTotal += 1
            else:
                xmin = max(xmin, 0)
                xmax = min(xmax, w)
                ymin = max(ymin, 0)
                ymax = min(ymax, h)
                crop_img = image[ymin:ymax, xmin:xmax].copy()
                leftpad = xpad*padsides[0]
                rightpad = xpad*padsides[1]
                toppad = ypad*padsides[2]
                bottompad = ypad*padsides[3]
                padded_img = cv2.copyMakeBorder(crop_img, top=toppad, bottom=bottompad, left=leftpad, right=rightpad, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
                dest_path = savingpath + areanames[i]
                os.makedirs(dest_path, exist_ok=True)
                finalPath = dest_path + picID + ".jpg"
                cv2.imwrite(finalPath, padded_img)
                badpics+=1
                print("Padded image No." + str(badpics) + ", " + picID)
                outputPicTotal += 1
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        os.makedirs(helperPath, exist_ok=True)
        finalPath = helperPath + picID + ".jpg"
        cv2.imwrite(finalPath, image)
print("Created " + str(outputPicTotal) + " pictures from " + str(inputPicTotal) + " aligned images with " + str(badpics) + " padded images")
