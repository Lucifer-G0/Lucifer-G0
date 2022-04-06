#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/common/file_io.h> // for getFilenameWithoutExtension
#include <pcl/console/parse.h>
#include <pcl/segmentation/extract_clusters.h> // Euclidean Cluster Extract

#include <iostream>
#include <opencv2/opencv.hpp>

#include "ObjectWindow.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointWithRange PointWR;



using namespace std;

// --------------------
// -----Parameters-----
// --------------------
float angular_resolution = 0.5f;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool setUnseenToMaxRange = false;

float ec_cluster_tolerance=1.0f;

cv::Mat border2png(pcl::PointCloud<PointWR>::Ptr border_points_ptr,string filename="../res/00000-color.png");
cv::Mat border2png2(pcl::PointCloud<PointWR>::Ptr border_points_ptr,string filename="../res/00000-color.png");
void cluster_draw(string filename );


int main(int argc, char ** argv)
{
    // --------------------------------------
    // -----Parse Command Line Arguments-----
    // --------------------------------------
    if (pcl::console::parse (argc, argv, "-t", ec_cluster_tolerance) >= 0)
        std::cout << "Setting ec_cluster_tolerance to "<<ec_cluster_tolerance<<"deg.\n";

    pcl::PCDWriter writer;

    PointCloudT::Ptr cloud (new PointCloudT);
    PointCloudT::Ptr filtered_cloud (new PointCloudT);
    pcl::PointCloud<PointT>& point_cloud = *filtered_cloud;

    // load cloud from pcd file
    pcl::console::print_highlight ("Loading point cloud...\n");
    if (pcl::io::loadPCDFile<PointT> (argv[1], *cloud))
    {
        pcl::console::print_error ("Error loading cloud file!\n");
        return (1);
    }

    // passthrough filter, remove 0
    pcl::PassThrough<PointT> pass;

    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (1, 50);   // test pcd: Most points range from 7 to 30.
    pass.filter (*filtered_cloud);
    std::cerr << "PointCloud after filtering has: " << filtered_cloud->size () << " data points." << std::endl;

    // border extract
    pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;

    //!!!!!!!!!! very very important
    angular_resolution = pcl::deg2rad (angular_resolution);

    Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());

    // maybe useless for me. For test pcd, its sensor_origin is 0 0 0
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (point_cloud.sensor_origin_[0],
                                                               point_cloud.sensor_origin_[1],
                                                               point_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (point_cloud.sensor_orientation_);

    std::string far_ranges_filename = pcl::getFilenameWithoutExtension (argv[1])+"_far_ranges.pcd";
    if (pcl::io::loadPCDFile(far_ranges_filename.c_str(), far_ranges) == -1)
      std::cout << "Far ranges file \""<<far_ranges_filename<<"\" does not exists.\n";
    


    // -----Create RangeImage from the PointCloud-----
    float noise_level = 0.0;
    float min_range = 0.0f;
    int border_size = 1;
    pcl::RangeImage::Ptr range_image_ptr (new pcl::RangeImage);
    pcl::RangeImage& range_image = *range_image_ptr;
    range_image.createFromPointCloud (point_cloud, angular_resolution, pcl::deg2rad (360.0f), pcl::deg2rad (180.0f),
                                   scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
    range_image.integrateFarRanges (far_ranges);
    range_image.setUnseenToMaxRange ();

    // -----Open 3D viewer and add point cloud-----
    pcl::visualization::PCLVisualizer viewer ("3D Viewer");
    viewer.setBackgroundColor (1, 1, 1);    // white
    // viewer.addCoordinateSystem (1.0f, "global");
    pcl::visualization::PointCloudColorHandlerCustom<PointT> point_cloud_color_handler (filtered_cloud, 0, 0, 0);
    viewer.addPointCloud (filtered_cloud, point_cloud_color_handler, "original point cloud");

     // -----Extract borders-----
	pcl::RangeImageBorderExtractor border_extractor (&range_image);
  pcl::PointCloud<pcl::BorderDescription> border_descriptions;
  border_extractor.getAnglesImageForBorderDirections();
	border_extractor.compute (border_descriptions);

	// ----------------------------------
	// -----Show points in 3D viewer-----
	// ----------------------------------
	pcl::PointCloud<PointWR>::Ptr border_points_ptr(new pcl::PointCloud<PointWR>),
	                                          veil_points_ptr(new pcl::PointCloud<PointWR>),
                                            shadow_points_ptr(new pcl::PointCloud<PointWR>);
    pcl::PointCloud<PointWR>& border_points = *border_points_ptr,
                                      & veil_points = * veil_points_ptr,
                                      & shadow_points = *shadow_points_ptr;
    for (int y=0; y< (int)range_image.height; ++y)
	{
		for (int x=0; x< (int)range_image.width; ++x)
		{
			if (border_descriptions[y*range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER])
				border_points.push_back (range_image[y*range_image.width + x]);
			if (border_descriptions[y*range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT])
				veil_points.push_back (range_image[y*range_image.width + x]);
			if (border_descriptions[y*range_image.width + x].traits[pcl::BORDER_TRAIT__SHADOW_BORDER])
				shadow_points.push_back (range_image[y*range_image.width + x]);
		}
	}
	pcl::visualization::PointCloudColorHandlerCustom<PointWR> border_points_color_handler (border_points_ptr, 0, 255, 0);
	viewer.addPointCloud<PointWR> (border_points_ptr, border_points_color_handler, "border points");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");

	cout<<"border_points_ptr has "<<border_points_ptr->size()<<" points"<<endl;
  writer.writeASCII("borer_points.pcd",*border_points_ptr);

  
  
  cv::Mat image;
  image=border2png2(border_points_ptr);
  cv::imshow("border test",image);
  cv::waitKey(0);

	// while (!viewer.wasStopped ())
	// {
	// 	// range_image_borders_widget->spinOnce ();
	// 	viewer.spinOnce ();
	// 	pcl_sleep(0.01);
	// }


}

/*
    @brief  border to png
*/
cv::Mat border2png(pcl::PointCloud<PointWR>::Ptr border_points_ptr,string filename)
{
  cv::Mat image;
  image = cv::imread( filename, 1 );

  float constant = 570.3;

  for (auto& border_point: *border_points_ptr)
  {
    cv::Point point;//特征点，用以画在图像中  
	  point.x = border_point.x * constant / border_point.z + 320; // grid_x = x * constant / depth
	  point.y = border_point.y * constant / border_point.z + 240;
    cv::circle(image, point, 1, cv::Scalar(0, 0, 255));
  }
  
  return image;
}

/*
    @brief  border to png, add cluster extract
*/
cv::Mat border2png2(pcl::PointCloud<PointWR>::Ptr border_points_ptr,string filename)
{
  // Euclidean Cluster Extract

  pcl::search::KdTree<PointWR>::Ptr tree (new pcl::search::KdTree<PointWR>);
  tree->setInputCloud (border_points_ptr);
  cout<<"border_points_ptr has "<<border_points_ptr->size()<<" points"<<endl;

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointWR> ec;
  ec.setClusterTolerance (1);
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100);
  ec.setSearchMethod (tree);
  ec.setInputCloud (border_points_ptr);
  ec.extract (cluster_indices);

  // read png image and set constant
  cv::Mat image;
  image = cv::imread( filename, 1 );
  float constant = 570.3;

  //存储所有检测到的物体窗口，或者可以取消他在循环里面直接画到图上。
  std::vector<ObjectWindow> object_windows;
  pcl::PCDWriter writer;

  int j = 0;
  // 外循环 循环所有聚类
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {

    ObjectWindow object_window;

    pcl::PointCloud<PointWR>::Ptr cloud_cluster (new pcl::PointCloud<PointWR>);
    //内循环 循环一个聚类内部的所有点
    for (const auto& idx : it->indices)
    {
        PointWR border_point=(*border_points_ptr)[idx];
        cloud_cluster->push_back (border_point); //*
        object_window.add_point(border_point);
    }
      
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    object_window.update();
    object_window.output();
    image=object_window.draw(image);
    object_windows.push_back(object_window);

    // save cluster to pcd 
    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<PointWR> (ss.str (), *cloud_cluster, false); //*
    j++;
  }

  return image;
}

void cluster_draw(string filename )
{
  int j=0; // total cluster num, need change

  std::stringstream ss;
  ss<<"pcl_viewer -cam "<<pcl::getFilenameWithoutExtension (filename)<<".cam ";
  for(int i=0;i<j;i++)
  {
    ss << " -ps 20  cloud_cluster_" << i << ".pcd";
  }
  ss<<" -ps 1 "<<filename;
  std::cerr<<ss.str()<<std::endl;
  system(ss.str().c_str());
}