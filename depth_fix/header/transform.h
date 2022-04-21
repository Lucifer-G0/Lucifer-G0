#pragma once
#include<iostream>
#include "MyCloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

typedef pcl::PointXYZ PointT;
/*
	@brief  transform from a depth png to cloud : recommend use
	@param filename:	the path or name of depth pnd file, use for imread(filename,-1)
	@param test:	if you need save cloud to pcd and output some info (usually use for test,save name:filename_cloud.pcd)
	@return: cloud ptr
*/
pcl::PointCloud<PointT>::Ptr depth2cloud(std::string filename, bool test=false);

pcl::PointCloud<PointT>::Ptr depth2cloud_1(cv::Mat Depth);

pcl::PointCloud<PointT>::Ptr passthrough_filter(pcl::PointCloud<PointT>::Ptr, bool test = false);

MyCloud depth2cloud_2(std::string filename);