# FacialProcessing
FaceDetector.py
>>>Code to detect faces in an image using landmarks and a trained DNN from dlib. Using the LFW3D dataset. Outputs 64x64 images of faces from photos
PatchExtractor.py
>>>Processes 64x64 images from FaceDetector. Crops 6 images from each face, one centered on left/right eye, left/right eyebrow, mouth, and nose. Deposits into specified directory then performs HE illumination equalization on the photos. Final output is folders of greyscale features.

Sample input/output is from LFW3D dataset. Output_FaceDetector folder is the input for PatchExtractor
Trained DNN for shape prediction can be found here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
Trained DNN for face detection can be found here: http://dlib.net/files/mmod_human_face_detector.dat.bz2
