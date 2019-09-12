# FacialProcessing
Code written for the Media Communications lab at USC studying facial recognition using non-back propagation trained neural nets. Included code performs face detection in photos and the extraction of patches around 6 important facial landmarks. Code for neural net training not available at this time.

FaceDetector.py
  Code to detect faces in an image using landmarks and a trained DNN from dlib. Using the LFW3D dataset. Outputs 64x64 images of faces from photos
PatchExtractor.py
  Processes 64x64 images from FaceDetector. Crops 6 images from each face, one centered on left/right eye, left/right eyebrow, mouth, and nose. Deposits into specified directory then performs HE illumination equalization on the photos. Final output is folders of greyscale features.
PatchExtractorSmall.py
  Extracts a smaller patch size 20x20. On the LFW dataset, this code found that only eyebrows and mouth appear within 10 pixels of the bounds of the image. Therefore, this code takes patches centered 10 below eyebrows to ensure the image remains in bounds. Some eyebrows were actually in negative y values so in those cases, the would-be out of bounds areas are padded black to make a valid image. Eyes and nose patches are centered on feature. Mouth patches are centered 7 pixels above mouth feature.

Sample input/output is from LFW3D dataset. Output_FaceDetector folder is the input for PatchExtractor

Trained DNN for shape prediction can be found here: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Trained DNN for face detection can be found here: http://dlib.net/files/mmod_human_face_detector.dat.bz2

Example folders not accurate to current version of program output but chosen for easier human readability. Currently, output folders of PatchExtractor will be left_eyebrows, left_eyes, right_eyebrows, right_eyes, noses, mouths rather than sorted by name.
