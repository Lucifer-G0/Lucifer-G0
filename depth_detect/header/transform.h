#pragma once
#include<iostream>
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

pcl::PointCloud<PointT>::Ptr mat2cloud(cv::Mat Depth);

cv::Mat depth_to_uint8(cv::Mat depth);

int png2video(std::string img_path);