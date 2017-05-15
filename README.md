# Yolo version 2 for ROS
Yolo9000 wrapped in a ros-package for running it with ROS. 
A lot of credit goes to Joseph Chet Redmon - you might enjoy his [resume](https://pjreddie.com/static/Redmon%20Resume.pdf) - and co. for implementing a deep learning framework [Darknet](http://pjreddie.com/darknet/) and the super fast object detector [YOLO](https://arxiv.org/abs/1506.02640) and [YOLOv2 / YOLO9000](https://pjreddie.com/media/files/papers/YOLO9000.pdf).

The make-file have been converted to a CMakeLists.
(Code test on ubuntu 16.04, cuda 8, opencv 3.2.0-dev, ROS kinetic)

### Install ROS and make workspace
Follow the ROS [installation guide](http://wiki.ros.org/ROS/Installation) for download and installation and ROS [tutorials](http://wiki.ros.org/ROS/Tutorials) for creating a workspace. 

### Place package in your workspace src-folder
Go to ros-workspace source folder in a terminal

	cd [workspace]/src/

Clone folder in terminal.

	The repository needs to be updated. 
	$ git clone https://github.com/PeteHeine/yolo_v2_ros
	

Download yolo model to directory the weight directory.

	cd [workspace]/src/yolo_v2_ros
	mkdir weights
	cd weigths
	wget http://pjreddie.com/media/files/yolo.weights

Test yolo without ROS, without opencv and without GPU (In Makefile, set GPU=0 and OPENCV=0 and make.)
	
	cd [workspace]/src/yolo_v2_ros
	make
	./darknet detect cfg/yolo.cfg weights/yolo.weights data/dog.jpg

To run with GPU and opencv. Open Makefile, set GPU=1 and OPENCV=1 and make agian. 

### Build project with ROS
First clone bounding box message type in workspace. 

	cd [workspace]/src/
	git clone https://github.com/PeteHeine/msg_boundingbox.git
	
Open [workspace]/src/yolo_v2_ros/CMakeList.txt and ensure that

	set(OPENCV 1)
	set(ROS_INTERFACE 1) 


#### For CPU 
Open [workspace]/src/yolo_v2_ros/CMakeList.txt and ensure that

	set(GPU 0)

#### For GPU
Software requires CUDA to be installed. 
Go to [download page](https://developer.nvidia.com/cuda-downloads) select your platform, download and follow instructions.

Open [workspace]/src/yolo_v2_ros/CMakeList.txt and ensure that

	set(GPU 1)
#### Build

Go to workspace and run catkin make. 

	cd [workspace]
	catkin_make


### Test with ROS and usb cam
Clone usb_cam ros package to workspace.
Go again to ros-workspace source folder.

	cd [workspace]/src/

Clone folder

	git clone https://github.com/bosch-ros-pkg/usb_cam.git

Move file to weight-folder

	cd [workspace]/src/yolo_v2_ros
	mkdir weights

	# Normal model
	cp ~/Downloads/yolo.weights ./weights 


Build packages

	cd [workspace]
	catkin_make

Source ros-workspace

	source devel/setup.bash

Run launch file 

	roslaunch yolo_v2_ros usb_web_cam.launch



