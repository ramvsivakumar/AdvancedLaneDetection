# AdvancedLaneDetection
This project was inspired by the author 'Moataz-E'. I thank the author for this opportunity and this project would not have been possible without their github repo.

I wanted to learn the concepts used in Computer Vision for application development. While this script is effective in detecting road lanes, a deeper interpretaion on developing algorithms based on preprocessing, environment factors and sensors remains to be investigated.

Brief overview of this project:

This project is composed of:
  
  1. Camera Calibration using Chessboard images - 
    To account for radial distortion in images. The output images will be undistorted.
    
  2. Preprocessing the images for Perspective transformation - 
    Used a combination of two color channels, namely, LUV and LAB.
  
  3. Perspective transformation of images - 
    Used for the conversion of preprocessed and undistorted frames into a perspective view of the real word.
  
  4. Lane detection - 
  This method makes use of perspective view of the image with regards to its binary transformation. A sliding window is applied around nonzero pixels of the image representing    the lanes. The position of the window changes based on the mean value of the pixels. After this, a smooth line is generated around the mean of the pixels through a second order polynomial function. The area in the vicinity of between the polynomial lines is filled with a color, representing the detected lane.
    
  5. Filling polygon to the detected lanes with regards to the original frame.
  
