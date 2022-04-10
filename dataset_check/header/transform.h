#pragma once
#include<iostream>
#include "MyCloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

/*
	@brief  transform from a depth png to cloud : recommend use
	@param filename:	the path or name of depth pnd file, use for imread(filename,-1)
	@param test:	if you need save cloud to pcd and output some info (usually use for test,save name:filename_cloud.pcd)
	@return: cloud ptr
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr depth2cloud(std::string filename, bool test=false);

pcl::PointCloud<pcl::PointXYZ>::Ptr passthrough_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr, bool test = false);

MyCloud depth2cloud_2(std::string filename);