cmake_minimum_required(VERSION 2.8.3)
project(yolo9000ros)

#set(DEBUG 0) # Not implemented.
set(GPU 1)
set(OPENCV 1)
set(ROS_INTERFACE 1) # THIS REQUIRES OPENCV


set(COMMON_FLAGS "")
if (GPU)
	set(COMMON_FLAGS "${COMMON_FLAGS} -DGPU" )
	find_package(CUDA REQUIRED)
endif()

if (OPENCV)
	set(COMMON_FLAGS "${COMMON_FLAGS} -DOPENCV" )
endif()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
  camera_info_manager
  #htf_safe_msgs
)

find_package(OpenCV REQUIRED)

find_package(BLAS)
find_package(Threads)


#file(GLOB C_FILES
#    "src/*.h"
#    "src/*.c"
#)

set(CMAKE_C_FLAGS "-c -g -Ofast -Wall -Wfatal-errors ${COMMON_FLAGS}") # 

set(CUDA_NVCC_FLAGS 
	#${CUDA_NVCC_FLAGS}; This created an error and is therefore removed!!
	-O3
	${COMMON_FLAGS}
	)


##################################### Only for print out stuff #######################################################
if (OpenCV_FOUND)
	message("OPENCV found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 	
else()
	message("OPENCV not found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
endif()

if (CUDA_FOUND)
	message("CUDA found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 	#set(COMMON_FLAGS "${COMMON_FLAGS} -DGPU")
else()
	message("CUDA not found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
endif()
message("COMMON_FLAGS: " ${COMMON_FLAGS})
message("CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
message("CUDA_curand_LIBRARY: " ${CUDA_curand_LIBRARY})
#message("OpenCV_LIBS: " ${OpenCV_LIBS})
#message("catkin_LIBRARIES: " ${catkin_LIBRARIES})
#message("CMAKE_THREAD_LIBS_INIT: ",${CMAKE_THREAD_LIBS_INIT})
message("CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
message("CUDA_CUBLAS_LIBRARIES: " ${CUDA_CUBLAS_LIBRARIES})
message("CUDA_curand_LIBRARY: " ${CUDA_curand_LIBRARY})
# INCLUDES
#message("catkin_INCLUDE_DIRS: " ${catkin_INCLUDE_DIRS})
#message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
#########################################################################################################################


set(SOURCES_C 
	src/activations.c
	src/activation_layer.c
	src/avgpool_layer.c
	src/art.c
	src/batchnorm_layer.c
	src/blas.c
	src/box.c
	src/captcha.c
	src/classifier.c
	src/cifar.c
	src/coco.c
	src/col2im.c
	src/compare.c
	src/connected_layer.c
	src/convolutional_layer.c
	src/cost_layer.c
	src/crop_layer.c
	src/crnn_layer.c
	src/cuda.c
	src/data.c
	src/demo.c
	src/deconvolutional_layer.c
	src/detection_layer.c
	src/detector.c
	src/dice.c
	src/dropout_layer.c
	src/gemm.c
	src/go.c
	src/gru_layer.c
	src/im2col.c
	src/image.c
	src/layer.c
	src/list.c
	src/local_layer.c
	src/lsd.c
	src/matrix.c
	src/maxpool_layer.c
	src/network.c
	src/nightmare.c
	src/normalization_layer.c
	src/option_list.c
	src/parser.c
	src/region_layer.c
	src/regressor.c
	src/reorg_layer.c
	src/rnn.c
	src/rnn_layer.c
	src/rnn_vid.c
	src/route_layer.c
	src/shortcut_layer.c
	src/softmax_layer.c
	src/super.c
	src/swag.c
	src/tag.c
	src/tree.c
	src/utils.c
	src/voxel.c
	src/writing.c
	src/yolo.c
	src/yoloInterface.c
	) # "disinclude" src/imagenet.c, cpu_gemm, server, stb_image

set(SOURCES_CU 
	src/convolutional_kernels.cu
	src/deconvolutional_kernels.cu
	src/activation_kernels.cu
	src/im2col_kernels.cu
	src/col2im_kernels.cu
	src/blas_kernels.cu
	src/crop_layer_kernels.cu
	src/dropout_layer_kernels.cu
	src/maxpool_layer_kernels.cu
	src/network_kernels.cu
	src/avgpool_layer_kernels.cu) # disinclude src/softmax_layer_kernels.cu, src/yolo_kernels.cu, src/coco_kernels.cu

set(HEADER_FILES
	src/activations.h
	src/activation_layer.h
	src/avgpool_layer.h
	src/batchnorm_layer.h
	src/blas.h
	src/box.h
	src/col2im.h
	src/connected_layer.h
	src/convolutional_layer.h
	src/cost_layer.h
	src/crop_layer.h
	src/crnn_layer.h
	src/cuda.h
	src/data.h
	src/demo.h
	src/deconvolutional_layer.h
	src/detection_layer.h
	src/dropout_layer.h
	src/gemm.h
	src/im2col.h
	src/image.h
	src/layer.h
	src/list.h
	src/local_layer.h
	src/matrix.h
	src/maxpool_layer.h
	src/network.h
	src/normalization_layer.h
	src/option_list.h
	src/parser.h
	src/rnn_layer.h
	src/region_layer.h
	src/reorg_layer.h
	src/route_layer.h
	src/shortcut_layer.h
	src/softmax_layer.h
	src/tree.h
	src/utils.h
	src/yoloInterface.h
	) # stb_image.h, stb_image_write.h

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES darknet_ros
#  CATKIN_DEPENDS rosc roscpp std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  #${CUDA_INCLUDE_DIRS}  automatically included when using cuda_add_executable
)

## Declare a C++ library
# add_library(darknet_ros
#   src/${PROJECT_NAME}/dar-lcudart
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure



# Declare a C++ executable
#add_executable(darknet_ros_node src/yoloClass.cpp ${C_FILES})

if(GPU)
	set(ALL_FILES ${HEADER_FILES} ${SOURCES_C} ${SOURCES_CU})
else()
	set(ALL_FILES ${HEADER_FILES} ${SOURCES_C})
endif()

if(ROS_INTERFACE)

	if(GPU)
		cuda_add_executable(yolo9000ros_node src/yoloClass.cpp ${ALL_FILES} )
	else()
		add_executable(yolo9000ros_node src/yoloClass.cpp ${ALL_FILES} )
	endif()

	add_dependencies(yolo9000ros_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} htf_safe_msgs_gencpp)
	# Specify libraries to link a library or executable target against
	target_link_libraries(yolo9000ros_node
	  ${OpenCV_LIBS}
	  ${catkin_LIBRARIES}
	  ${CMAKE_THREAD_LIBS_INIT}
	  ${CUDA_LIBRARIES}
	  ${CUDA_CUBLAS_LIBRARIES}
	  ${CUDA_curand_LIBRARY}
	)
endif()

if(GPU)
	cuda_add_executable(darknet src/darknet.c ${ALL_FILES} )
else()
	add_executable(darknet src/darknet.c ${ALL_FILES} )
endif()
# Specify libraries to link a library or executable target against
target_link_libraries(darknet
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CUDA_curand_LIBRARY}
)
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS darknet_ros darknet_ros_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_darknet_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
