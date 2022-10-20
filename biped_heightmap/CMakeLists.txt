cmake_minimum_required(VERSION 3.0.2)
project(biped_heightmap)

add_compile_options(-std=c++11)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  biped_heightmap_msgs
  jsk_pcl_ros
  jsk_recognition_msgs
  sensor_msgs
  cv_bridge
  image_transport
  roscpp
  pluginlib
  nodelet
  jsk_topic_tools
  )

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
)

add_library(${PROJECT_NAME} src/heightmap_estimator.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
