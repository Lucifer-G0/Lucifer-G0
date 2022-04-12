cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(test_NRLC)

find_package(PCL  1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(test_NRLC test_NRLC.cpp normal_estimation.cpp)

target_link_libraries(test_NRLC ${PCL_LIBRARIES})
