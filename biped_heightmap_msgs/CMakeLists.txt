cmake_minimum_required(VERSION 3.0.2)
project(biped_heightmap_msgs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  std_msgs
  geometry_msgs
  )

add_message_files(
  FILES
  SteppableRegion.msg
  LandingPosition.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES biped_heightmap_msgs
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)
