#! /bin/bash
echo "project name: $1"

# add project to the CMakeLists.txt of main project
sed -i "6i $1" CMakeLists.txt

# create project dir and create cpp and build directory
mkdir $1
cd $1
touch "$1.cpp"
cppfile="$1.cpp"
echo "">>$cppfile
echo "">>$cppfile
echo "int main()">>$cppfile
echo "{">>$cppfile
echo "">>$cppfile
echo "">>$cppfile
echo "}">>$cppfile

mkdir build

# create CMakeLists.txt init with pcl
fname="CMakeLists.txt"
touch $fname
echo "cmake_minimum_required(VERSION 3.5 FATAL_ERROR)">>$fname
echo "">>$fname
echo "project($1)">>$fname
echo "">>$fname
echo "find_package(PCL 1.8 REQUIRED)">>$fname
echo "">>$fname
echo "include_directories(\${PCL_INCLUDE_DIRS})">>$fname
echo "link_directories(\${PCL_LIBRARY_DIRS})">>$fname
echo "add_definitions(\${PCL_DEFINITIONS})">>$fname
echo "">>$fname
echo "add_executable($1 $1.cpp)">>$fname
echo "">>$fname
echo "target_link_libraries($1 \${PCL_LIBRARIES})">>$fname
cat $fname

