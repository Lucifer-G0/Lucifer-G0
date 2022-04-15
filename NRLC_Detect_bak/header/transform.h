#pragma once
#include<iostream>
#include "MyCloud.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

/*
	@brief  transform from a depth png to cloud : recommend use
	@param filename:	the path or name of depth pnd file, use for imread(filename,-1)
	@param test:	if you need save cloud to pcd and output some info (usually use for test,save name:filename_cloud.pcd)
	@return cloud ptr
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr depth2cloud(std::string filename, bool test=false);

/*
	@brief 直通滤波器，过滤掉数据缺失导致的深度为0的数据。
	@param cloud: 需要过滤的数据
	@param min: 保留的最小值
	@param min: 保留的最大值
	@param test: 是否测试模式，主要表现为是否输出保存结果为pcd
	@return cloud_filtered:过滤后的数据
*/
pcl::PointCloud<pcl::PointXYZ>::Ptr passthrough_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float min=1, float max=50,bool test = false);

MyCloud depth2cloud_2(std::string filename);