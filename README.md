# AugmentedRealityCPP
Augmented reality implemented in C++ using OpenCV for plane estimation and OpenGL for rendering graphics.

To switch between the two versions (2D planar augmented reality and 3D augmented reality, edit CMakeLists.txt executable accordingly. To compile the program, 
stand in the build directory and type "make" to get an executable named "chapter1" in the same directory.

--------------------------

augmented.cc - Finds a square object in the scene using thresholding + contours, creates a tracker of found area and looks for a square object using more robust 
thresholding in the tracked area. Once an object is found, 4 corners are extracted, sorted and used to create a homography from any image corners, into the scene. 
The image is then transformed and rendered into the scene using a mask and bitwise operations.

augmentedObject.cc - Similar to augmented.cc except that it also implements openGL 3D-rendering of a cube. This is done by extracting all the relevant info
from the image using same approach as in augmented.cc, but then saving the openCV-frame as a texture, rendering it using two triangles that cover the screen, then
adding the cube using the information from the openCV-info previously extracted.
