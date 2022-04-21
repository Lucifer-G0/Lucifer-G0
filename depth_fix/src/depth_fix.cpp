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
#include "Fix.h"
typedef pcl::PointXYZ PointT;
cv::Mat depth_to_uint8(cv::Mat depth);
int main()
{
	cv::Mat Depth = cv::imread("../scene_01/00000-depth.png", -1);
	Depth.convertTo(Depth, CV_32F);
	Fix fix(Depth);

	// All the objects needed
	pcl::PassThrough<PointT> pass;
	pcl::PCDWriter writer;
	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	// Datasets
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr back_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.2);

	cloud = depth2cloud("../scene_01/00000-depth.png");

	//-----------------------------根据阈值过滤出背景--------------------------------------------------
	// Build a passthrough filter to remove spurious NaNs and scene background
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(20, 50);
	pass.filter(*cloud_filtered);
	std::cerr << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

	//----------------------对背景聚类---------------------------------------------------------------
	tree->setInputCloud(cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(0.5);
	ec.setMinClusterSize(300);
	ec.setMaxClusterSize(20000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud_filtered);
	ec.extract(cluster_indices);

	if(cluster_indices.size()==0)
	{
		std::cout<<"cluster_indices.size()==0"<<std::endl;
		return 1;
	}
	else
	{
		std::cout<<cluster_indices.size()<<std::endl;
	}

	int j = 0;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (const auto &idx : it->indices)
			cloud_cluster->push_back((*cloud_filtered)[idx]); //*
		cloud_cluster->width = cloud_cluster->size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		//----write cluster to pcd --------------
		std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
		std::stringstream ss;
		ss << "cloud_cluster_" << j << ".pcd";
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false); //*

		//---- Segment the largest planar component from the remaining cloud---------------
		seg.setInputCloud(cloud_cluster);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		fix.back_plane_fix(cloud_cluster, inliers,coefficients);

		j++;
	}

	Depth = depth_to_uint8(fix.get_result());

	cv::imwrite("fixed_depth.png", Depth);

	back_cloud = depth2cloud_1(fix.get_result());
	pcl::io::savePCDFile("fixed_depth_cloud.pcd", *back_cloud);

	return 0;
}

cv::Mat depth_to_uint8(cv::Mat depth)
{
	cv::Mat Depth(depth.rows, depth.cols, CV_8U);

	float max = 0;
	for (int r = 0; r < depth.rows; r++)
	{
		//列遍历
		for (int c = 0; c < depth.cols; c++)
		{
			if (depth.at<float>(r, c) > max)
				max = depth.at<float>(r, c);
		}
	}
	for (int r = 0; r < depth.rows; r++)
	{
		//列遍历
		for (int c = 0; c < depth.cols; c++)
		{
			Depth.at<uchar>(r, c) = 255 * (depth.at<float>(r, c) / max);
		}
	}
	// cv::imshow("fix-depth",Depth);
	// cv::waitKey();
	return Depth;
}

void cluster_filter()
{
	// All the objects needed
	pcl::PassThrough<PointT> pass;
	pcl::PCDWriter writer;
	pcl::ExtractIndices<PointT> extract;
	pcl::ExtractIndices<pcl::Normal> extract_normals;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	// Datasets
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	cloud = depth2cloud("../scene_01/00000-depth.png");

	// Build a passthrough filter to remove spurious NaNs and scene background
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(1, 50);
	pass.filter(*cloud_filtered);
	std::cerr << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;

	// Create the segmentation object for the planar model and set all the parameters
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());

	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(0.02);

	int nr_points = (int)cloud_filtered->size();
	int i = 0;
	while (cloud_filtered->size() > 0.2 * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);

		// Get the points associated with the planar surface
		extract.filter(*cloud_plane);
		std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

		std::stringstream ss;
		ss << "planar_" << i++ << ".pcd";
		pcl::io::savePCDFile(ss.str(), *cloud_plane);

		// Remove the planar inliers, extract the rest
		extract.setNegative(true);
		extract.filter(*cloud_f);
		*cloud_filtered = *cloud_f;
	}
}