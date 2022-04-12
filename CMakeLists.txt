cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project("Depth_Object_Detect")

foreach(subdir
test_NRLC
test_cluster_blue
test_NRLC
border2png
	
)

add_subdirectory(${subdir})

endforeach()
