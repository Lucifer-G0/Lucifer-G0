#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;

class BackGround
{
public:
	cv::Mat Depth;
	

	int width;
	int height;
    float constant=570.3;
	cv::Mat seg_image=cv::Mat::zeros(480,640,CV_8U);    //分割结果存储图像，默认为0,先填背景，再填前景平面，再填物体

	BackGround(cv::Mat  _Depth);

	cv::Mat get_result(){return Depth;}
    void back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers,pcl::ModelCoefficients::Ptr coefficients);
	void back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
	void back_cluster_extract(pcl::PointCloud<PointT>::Ptr cloud_background,int dimension=2);
private:
	int vp_start_no=200;
	int vp_no;       //背景面的索引
	cv::Mat Mask;	//原始是否空洞点掩码，是空洞点为1，初始化时根据图像初始化，之后仅作为(多次填充)指示而不发生变化
	std::vector<float> min_depths;	//维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准
};
