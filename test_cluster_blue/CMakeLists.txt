cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(test_cluster_blue)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(test_cluster_blue test_cluster_blue.cpp ObjectWindow.cpp)

target_link_libraries(test_cluster_blue ${OpenCV_LIBS} ${PCL_LIBRARIES})
