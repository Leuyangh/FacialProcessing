# FacialProcessing
FaceDetector.py
>Code to detect faces in an image using landmarks and a trained DNN from dlib. Using the LFW3D dataset. Outputs 64x64 images of faces from photos
PatchExtractor.py
>Processes 64x64 images from FaceDetector. Crops 6 images from each face, one centered on left/right eye, left/right eyebrow, mouth, and nose. Deposits into specified directory then performs HE illumination equalization on the photos. Final output is folders of greyscale features.
