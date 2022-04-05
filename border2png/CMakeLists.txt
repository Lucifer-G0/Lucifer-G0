cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(border2png)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(border2png border2png.cpp)

target_link_libraries(border2png ${OpenCV_LIBS} ${PCL_LIBRARIES})
