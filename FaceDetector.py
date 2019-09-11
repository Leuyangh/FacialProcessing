# -*- coding: utf-8 -*-
"""

@author: Eric
"""

import argparse
from glob import glob
import cv2
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import dlib
import numpy as np
import os

"""
Face detection
"""

detector = dlib.get_frontal_face_detector()
dnnFaceDetector = dlib.cnn_face_detection_model_v1("C:\\Users\\Eric\\Documents\\Face_recognition_project\\mmod_human_face_detector.dat\\mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("C:\\Users\\Eric\\Documents\\Face_recognition_project\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat")
face_aligner = FaceAligner(predictor, desiredLeftEye=(0.23, 0.23), desiredFaceWidth=64, desiredFaceHeight=64)

datasetPath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\LFW3D.0.1.1\\LFW3D.0.1.1\\"
savingpath= "C:\\Users\\Eric\\Documents\\Face_recognition_project\\aligned3DLfw\\"
identities = glob(datasetPath+"*")
count=0
piccount=0
for identity in identities:
    count+=1
    identityStr = os.path.basename(identity).split('\\')[-1]
    pics=glob(identity+"\\*")
    print("creating pathes for identity " +identityStr)
    dest_path= savingpath + identityStr + "\\"
    os.makedirs(dest_path, exist_ok=True)
    for pic in pics:
        piccount+=1
        picId=os.path.basename(pic).split('\\')[-1]
        image = cv2.imread(pic)      
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = dnnFaceDetector(image, 0)
        # loop over the face detections
        for rect in rects:
            # extract the ROI of the *original* face, then align the face
            # using facial landmarks
            (x, y, w, h) = rect_to_bb(rect.rect)
            face_aligned = face_aligner.align(image, gray, rect.rect)
            cv2.imwrite(dest_path + picId, face_aligned)
print("TOTAL IMAGES: " + str(piccount))