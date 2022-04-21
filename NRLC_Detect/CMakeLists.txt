cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(NRLC_Detect)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_library(NRLC  ./header/NRLC.h ./header/NRLC.cpp)
set_target_properties(NRLC PROPERTIES LINKER_LANGUAGE CXX)

include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ./header )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

aux_source_directory(./src ALL_SRCS)    # 把当前目录下所有原文件添加到 ALL_SRCS（一个自定义变量）
add_executable(NRLC_Detect ${ALL_SRCS})  # 指定生成目标，注意所有 .o 都应该链接到目标

target_link_libraries(NRLC_Detect  ${OpenCV_LIBS} ${PCL_LIBRARIES} NRLC)

# set(CMAKE_BUILD_TYPE "Release" )
