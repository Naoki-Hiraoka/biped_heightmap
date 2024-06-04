cmake_minimum_required(VERSION 3.0.2)
project(biped_heightmap)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  jsk_pcl_ros
  jsk_topic_tools
  biped_heightmap_msgs
  jsk_recognition_msgs
  sensor_msgs
  cv_bridge
  image_transport
  roscpp
  pluginlib
  nodelet
  tf_conversions
  visualization_msgs
  )
find_package(OpenCV REQUIRED core imgproc)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES biped_heightmap
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
)

add_library(${PROJECT_NAME}
  src/heightmap_estimator.cpp
  src/steppable_region_publisher.cpp
  src/polypartition.cpp
  )
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  )
