----- README -----
This archive contains all the code used for developing and evaluating the FBOF algorithm, 
which is the proposed method from the master's dissertation 'Noise-robust motion detection for low-light videos' by Tim Ranson.

The 'Code' contains all the source code and binaries to build and run several applications which are listed below.
All applications were developed and tested using the OpenCV 3.1 C++ API library and require CMake 3.1 to build them.
For each application a bash shell script is provided (run_XXX) which compiles, builds and runs the application, provided the necessary libraries are installed.
For each application a CMakeLists_XXX.txt file is set up, which is copied to CMakeLists.txt by the openCvCmake.sh script in order to cmake the application.

----- APPLICATIONS -----
---addNoise---
Adds noise of various levels to the input dataset in 'input/' and saves the datasets with added noise to 'output/' in the structure used for the 'eval' and 'eval_fbof_parameters' applications.

---demo---
Runs a demo of the FBOF algorithm, showing the input and output alongside each other.
There are several switches available, which are documented at the bottom of the source code file 'src/demo.cpp'.
Mainly, pressing 'p' toggles play/pause and pressing 'enter' will continue to the next frame if in paused mode.

---eval---
This application runs the evaluation of the proposed method and the other algorithms to compare to.
Runs each algorithm and calculates the metrics discussed in the master's dissertation.

---eval_fbof_parameters---
This application runs the evaluation of the tunable parameters of the proposed method as discribed in the master's dissertation.

---main---
Runs the FBOF algorithm on a set dataset and shows intermediate results of the algorithm.
There are several switches available, which are documented at the bottom of the source code file 'src/main.cpp'.
Mainly, pressing 'p' toggles play/pause and pressing 'enter' will continue to the next frame if in paused mode.
The file 'FBOF_API.txt' explains the implementation of the MotionDetection::FBOF class in detail.

----- ESSENTIAL FILES ------
-- src/motion_detection --
src/motion_detection/fbof.h:
	Contains the C++ implementation of the FBOF algorithm. The code is documented, but a more thorough description can be found in 'FBOF_API.txt'.

src/motion_detection/motion_detection.h:
	Implements wrapper class for running the motion detection algorithms.

-- src/headers --
src/headers/dataset.h:
	Implements wrapper class for reading input datasets.
src/headers/denoising.h:
	Implements some helper functions for the FBOF algorithm.
src/headers/io.h:
	Contains all configurations and functions for IO operations (image displaying, file IO, ...). This file contains the DIR_INPUT and DIR_OUTPUT directories which are used for reading and writing all file IO to.
src/headers/scores.h:
	Implements calculations of the metrics discussed in the master's dissertation.

