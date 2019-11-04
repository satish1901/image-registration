# Image-Registration
Image Registration is the process of transforming two or more images/data into the same coordinate system. One of the images is referred as fixed/base image, is considered the reference to other input(moving) images. The following steps are used in the process of transforming, alignment and warpping the images:
1. Point Feature Detection
2. Establishing Correspondences 
3. Estimating the Homography
   - DLT
   - Normalized DLT
   - RANSAC
4. Image Warping 

## Required Libraries:
   1. OpenCv 3.4.2.16 (you can install this version using pip3)
   (You can use opencv 3.4.3 version too, but you will have to install it using CMAKE and set the flag NON_FREE_MODULE on) 
   2. python 3.5/3.6 
  
## How to Run Code:
   python img_reg.py -i image_1.png image_2.png -a "the method for image registration" 
   
   If you don't provide any method, it will run RANSAC bydefault
 
## How to install OpenCV 3.4.3 using CMAKE
You can follow this link
 https://docs.opencv.org/3.4.3/d7/d9f/tutorial_linux_install.html
 
 In Step2, you need to use the following command

CXXFLAGS="-std=c++11" cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_opencv_java=OFF \
-D OPENCV_EXTRA_MODULES_PATH="path to opencv_contrib-3.4.3/modules" \
-D WITH_TBB=ON \
-D WITH_OPENGL=ON \
-D PYTHON_EXECUTABLE=~/.virtualenvs/aero_cv/bin/python or "path to virtual environment python" \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=ON ..
