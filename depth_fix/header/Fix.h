#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;

class Fix
{
public:
	cv::Mat Depth;
	cv::Mat Mask;
	int width;
	int height;
    float constant=570.3;

	Fix(cv::Mat  _Depth);
	int get_right_c(int r,int c);
	int get_down_r(int r,int c);
	int fix(int r,int c);

	cv::Mat get_result(){return Depth;}
    void back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers,pcl::ModelCoefficients::Ptr coefficients);
};
