cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(data_check)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ./header)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

aux_source_directory(./src ALL_SRCS)    # 把当前目录下所有原文件添加到 ALL_SRCS（一个自定义变量）
add_executable(data_check ${ALL_SRCS})  # 指定生成目标，注意所有 .o 都应该链接到目标

target_link_libraries(data_check ${OpenCV_LIBS} ${PCL_LIBRARIES})
