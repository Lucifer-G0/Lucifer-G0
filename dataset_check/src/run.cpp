#include <thread>
#include<iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>

#include "normal_estimation.h"
#include "transform.h"

int NRLC_test(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals,float ft=0.85);
void normal_show(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud, pcl::PointCloud<pcl::Normal>::Ptr normals);
void cloud_show(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

int main(int argc, char *argv[])
{
		// load point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_f (new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_f_f (new pcl::PointCloud<pcl::Normal>);
	pcl::ExtractIndices<pcl::Normal> extract_normals_f;

	pcl::io::loadPCDFile("../res/00000_cloud.pcd", *cloud);
	std::cout << cloud->size() << std::endl;

	cloud_normals_f=fast_normal_estimation(cloud,true,"raw_40");

	std::cout<<"cloud_normals_f has "<<cloud_normals_f->size()<<std::endl;

	// passthrough filter, remove 0
    pcl::PassThrough<pcl::PointXYZ> pass(true);
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (1, 50);   // test pcd: Most points range from 7 to 30.
    pass.filter (*filtered_cloud);
	pcl::IndicesConstPtr inliers = pass.getRemovedIndices();
	std::cout<<"inliers has "<<inliers->size()<<std::endl;

    // pcl::io::savePCDFile("filtered_cloud.pcd", *filtered_cloud);
	// cout << "save filtered_cloud.pcd finish" << endl;

    fast_normal_estimation(filtered_cloud,true,"filtered_cloud_40");

	extract_normals_f.setNegative (true);
  	extract_normals_f.setInputCloud (cloud_normals_f);
  	extract_normals_f.setIndices (inliers);
  	extract_normals_f.filter (*cloud_normals_f_f);


	pcl::io::savePCDFile("filtered_raw_40_fnormals.pcd", *cloud_normals_f_f);


	return 0;
}





void store_useful_sentence()
{
	// std::string imgs_path="../imgs";
	// std::vector<cv::String> img;
	// cv::glob(imgs_path, img, true);
	// int img_num=img.size();
	// std::cout<<img_num<<std::endl;
	// pcl::console::TicToc tt;
	// pcl::console::TicToc tt2;
	// tt.tic();
	// tt2.tic();
	// int j=0;
	// for(int i=0;i<img_num;i++)
	// {
	// 	std::string filename=img[i];
	// 	depth2cloud(filename,true);
	// 	if(++j % 100 == 0)
	// 	{
	// 		std::cout<<"transfer 100 depth png to pcd cost:"<<tt2.toc()<<" ms."<<std::endl;
	// 		tt2.tic();
	// 	}
		
	// }
	// std::cout<<"transfer "<<img_num<<"depth png to pcd cost:"<<tt.toc()<<" ms."<<std::endl;


	// 	// load point cloud
	// pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	// pcl::io::loadPCDFile("object.pcd", *cloud);
	// std::cout << cloud->size() << std::endl;

	// //pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// //pcl::io::loadPCDFile("res/organized_normals.pcd", *normals);
	// //std::cout << normals->size() << std::endl;

	// cloud_show(cloud);

	//std::string sft = argv[1];

	//float ft;
	//std::stringstream stream(sft);
	//stream >> ft;
	//std::cout << ft << std::endl;
	//pcl::console::TicToc tt;
	//tt.tic(); 
	//NRLC_test(cloud, normals);
	//cout << "NRLC cost: " << tt.toc() << "ms" << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = depth2cloud("res/00000-depth.png");
	pcl::PointCloud<pcl::Normal>::Ptr normals = orgnized_normal_estimation(cloud);

	cloud_show(cloud);
	normal_show(cloud, normals);

	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = passthrough_filter(cloud, true);
	pcl::PointCloud<pcl::Normal>::Ptr normals_1 = fast_normal_estimation(filtered_cloud, true, "dense_fast_normals");
}


void cloud_show(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	pcl::visualization::PCLVisualizer visualizer("show result");
	visualizer.addPointCloud(cloud);
	while (!visualizer.wasStopped())
	{
		visualizer.spinOnce(100);
		pcl_sleep(0.01);
	}
	int a; cin >> a;
}

void normal_show(pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud, pcl::PointCloud<pcl::Normal>::Ptr normals)
{
	// visualize normals
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0.0, 0.0, 0.5);
	//viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 10, 0.05, "normals");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}