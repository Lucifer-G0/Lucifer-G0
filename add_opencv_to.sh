#! /bin/bash
echo "add opencv to project $1"
cd $1
fname=CMakeLists.txt
sed -i "6i find_package(OpenCV REQUIRED)" $fname
sed -i "s/include_directories(/include_directories(\${OpenCV_INCLUDE_DIRS} /" $fname
sed -i "s/target_link_libraries($1 /target_link_libraries($1 \${OpenCV_LIBS} /" $fname
cat $fname
