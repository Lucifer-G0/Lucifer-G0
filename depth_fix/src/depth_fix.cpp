#include <opencv2/opencv.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include "transform.h"
#include "ObjectWindow.h"
#include "BackGround.h"
#include "ForeGround.h"

typedef pcl::PointXYZ PointT;
cv::Mat depth_to_uint8(cv::Mat depth);

int main()
{
	float back_threshold_percent = 0.85f; //用于计算背景的深度阈值，百分比形式。0.85比较合适？
	float back_threshold = 0.0f;
	float max_depth = 50.0f;
	float fore_seg_threshold_percent = 0.1f; //前景分割是否平面阈值，前景点云大小的百分比

	std::string depth_path = "00000-depth.png";

	cv::Mat Depth = cv::imread(depth_path, -1);
	Depth.convertTo(Depth, CV_32F);
	BackGround back(Depth);

	// All the objects needed
	pcl::PassThrough<PointT> pass;

	// Datasets
	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_background(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr back_fixed_cloud(new pcl::PointCloud<PointT>);
	pcl::PointCloud<PointT>::Ptr cloud_foreground(new pcl::PointCloud<PointT>);

	cloud = depth2cloud(depth_path);

	//--------------计算背景的深度阈值----------------------
	std::vector<float> sorted_Depth;
	for (auto &point : *cloud)
	{
		sorted_Depth.push_back(point.z);
	}
	std::sort(sorted_Depth.begin(), sorted_Depth.end());
	back_threshold = sorted_Depth[(int)(sorted_Depth.size() * back_threshold_percent)]; //根据百分比计算得到阈值
	max_depth = sorted_Depth[sorted_Depth.size() - 1] + 0.001;							//获得最大值，不清楚过滤的开闭，因而加一点避免最大值被过滤

	//-----------------------------根据阈值过滤出背景,分割出背景和前景--------------------------------------------------
	// Build a passthrough filter to remove spurious NaNs and scene background
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(back_threshold, max_depth); //阈值到最大值
	pass.filter(*cloud_background);					 //背景点云
	pass.setFilterLimits(0.001, back_threshold - 0.001);
	pass.filter(*cloud_foreground); //前景点云,需注意前景必须去除零点，因为零点占相当大部分
	pcl::io::savePCDFile("cloud_foreground.pcd", *cloud_foreground);

	back.back_cluster_extract(cloud_background);
	// cv::Mat color;
	// cv::applyColorMap(back.seg_image,color,cv::COLORMAP_HSV);
	// cv::imshow("object_detect_2D", color);

	// while (cv::waitKey(100) != 27)
	// {
	// 	if (cv::getWindowProperty("object_detect_2D", 0) == -1) //处理手动点击叉号关闭退出，报错退出，只能放在结尾
	// 		break;
	// }

	// //---------------backgroud fix output---------------------------------
	// Depth = depth_to_uint8(back.get_result());
	// cv::imwrite("fixed_depth.png", Depth);
	// back_fixed_cloud = mat2cloud(back.get_result());
	// pcl::io::savePCDFile("fixed_depth_cloud.pcd", *back_fixed_cloud);

	// //---------------------------------前景平面修复,需注意前景必须去除零点，因为零点占相当大部分-----------------------------------------------------
	// ForeGround fore(cloud_foreground,fore_seg_threshold_percent);
	// fore.planar_seg();

	// // Depth = depth_to_uint8(fix.get_result());

	// // std::vector<cv::Point> border_points=fore.extract_border_2D(fore.plane_clouds[0]);
	// // for (auto &point : border_points)
	// // {
	// //     cv::circle(Depth,point,0.1,cv::Scalar(200));
	// // }
	// // cv::imshow("show 2D border",Depth);
	// // cv::imwrite("border_2D.png",Depth);
	// // cv::waitKey();

	// pcl::io::savePCDFile("fore_remove_support.pcd", *fore.cloud_foreground);
	// fore.border_clean();
	// fore.object_detect_2D();
	// cv::imshow("object_detect_2D",fore.seg_image);

	// while (cv::waitKey(100) != 27)
	// {
	// 	if(cv::getWindowProperty("object_detect_2D",0) == -1)//处理手动点击叉号关闭退出，报错退出，只能放在结尾
	// 		break;
	// }
	return 0;
}

void bak_code()
{
	// 记录起始的时钟周期数
	double time = (double)cv::getTickCount();

	// 计算时间差
	time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();

	// 输出运行时间
	std::cout << "运行时间：" << time << "秒\n";

	// std::cerr << "PointCloud after filtering has: " << cloud_background->size() << " data points." << std::endl;

	// //----write cluster to pcd --------------
	// std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
	// std::stringstream ss;
	// ss << "cloud_cluster_" << j << ".pcd";
	// pcl::io::savePCDFile(ss.str(), *cloud_cluster);
	// std::stringstream ss;
	// ss << "cloud_plane_" << i << ".pcd";
	// pcl::io::savePCDFile(ss.str(), *cloud_plane);
}