cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(depth_fix)

find_package(OpenCV REQUIRED )

find_package(PCL REQUIRED)
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
include_directories(${OpenCV_INCLUDE_DIRS}  ./header )

aux_source_directory(./src ALL_SRCS)    # 把当前目录下所有原文件添加到 ALL_SRCS（一个自定义变量）
add_executable(depth_detect ${ALL_SRCS})  # 指定生成目标，注意所有 .o 都应该链接到目标

target_link_libraries(depth_detect  ${OpenCV_LIBS}  ${PCL_LIBRARIES})

