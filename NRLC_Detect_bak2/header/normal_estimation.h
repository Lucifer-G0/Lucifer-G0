#pragma once
#include <pcl/io/pcd_io.h>
#include <iostream>
/*
	@brief  Normal Estimation Using Integral Images: recommend use
	@param cloud:	inputcloud
	@param test:	if you need save normals to pcd and output some info (usually use for test)
	@return: normals ptr
*/
pcl::PointCloud<pcl::Normal>::Ptr orgnized_normal_estimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool test=false,std::string name="organized");

/*
	@brief  Normal Estimation Using OpenMP,多线程，可以指定KSearch
	@param cloud:	inputcloud
	@param test:	if you need save normals to pcd and output some info (usually use for test)
	@param name:	pcd name(auto add suffix _fnormals.pcd)
	@return normals ptr
*/
pcl::PointCloud<pcl::Normal>::Ptr fast_normal_estimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, bool test = false, std::string name = "fast");

// single thread: long long time , no back ground (soft and so on, which is not useless)
pcl::PointCloud<pcl::Normal>::Ptr usual_normal_estimation(); 