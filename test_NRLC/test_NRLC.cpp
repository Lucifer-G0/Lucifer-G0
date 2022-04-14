#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/passthrough.h>

#include "NRLC.hpp"
#include "normal_estimation.h"
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;

int main()
{
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

    // load point cloud
	pcl::io::loadPCDFile("00000_cloud_filtered.pcd", *cloud_filtered);
	std::cout << cloud_filtered->size() << std::endl;
	// load point cloud
	pcl::io::loadPCDFile("00000_cloud_fnormals.pcd", *normals);
	std::cout << normals->size() << std::endl;


    NRLC nrlc;
	nrlc.setInputCloud(cloud_filtered);
	nrlc.setNormals(normals);
	nrlc.setParams(40, 60, 0.7, 0.85);
	std::vector<int> vec_n_feature;
	nrlc.detect(vec_n_feature);
	// nrlc.EDE(vec_n_feature, 4, 1, 1);
	// nrlc.refine(3, vec_n_feature);
 
	// visualization
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr visual_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (size_t i = 0; i < cloud_filtered->size(); i++)
	{
        // std::cout<<vec_n_feature[i]<<std::endl;
		pcl::PointXYZRGB point;
		point.x = cloud_filtered->points[i].x;
		point.y = cloud_filtered->points[i].y;
		point.z = cloud_filtered->points[i].z;
		if (vec_n_feature[i] == 1) // convex
		{
			point.r = 0;
			point.g = 0;
			point.b = 255;
			visual_cloud->push_back(point);
		}
		else if (vec_n_feature[i] == 2) // concave
		{
			point.r = 255;
			point.g = 0;
			point.b = 0;
			visual_cloud->push_back(point);
		}
		else if (vec_n_feature[i] == 3) // border
		{
			point.r = 0;
			point.g = 255;
			point.b = 0;
			visual_cloud->push_back(point);
		}
		else if (vec_n_feature[i] == 0) // nonfeature
		{
			point.r = 150;
			point.g = 150;
			point.b = 150;
		}
	}
	pcl::io::savePCDFile("nrlc_test_out.pcd", *visual_cloud);
	cout << "save finish" << endl;

	pcl::visualization::PCLVisualizer visualizer("show result");
	visualizer.addPointCloud(visual_cloud);
	visualizer.spin();
    while (!visualizer.wasStopped())
	{
		visualizer.spinOnce(100);
		pcl_sleep(0.01);
	}

}

void store()
{
	  // // passthrough filter, remove 0
    // pcl::PassThrough<PointT> pass;
    // pass.setInputCloud (cloud);
    // pass.setFilterFieldName ("z");
    // pass.setFilterLimits (1, 50);   // test pcd: Most points range from 7 to 30.
    // pass.filter (*cloud_filtered);

    // pcl::io::savePCDFile("cloud_filtered.pcd", *cloud_filtered);
	// cout << "save finish" << endl;

    // normals=fast_normal_estimation(cloud_filtered,"00");
}
