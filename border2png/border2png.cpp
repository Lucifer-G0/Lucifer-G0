#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/common/file_io.h> // for getFilenameWithoutExtension

#include <iostream>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

using namespace std;

// --------------------
// -----Parameters-----
// --------------------
float angular_resolution = 0.5f;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool setUnseenToMaxRange = false;


int main(int argc, char ** argv)
{
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
	pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
	                                          veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
                                            shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
    pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr,
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
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> border_points_color_handler (border_points_ptr, 0, 255, 0);
	viewer.addPointCloud<pcl::PointWithRange> (border_points_ptr, border_points_color_handler, "border points");
	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "border points");

	cout<<"border_points_ptr has "<<border_points_ptr->size()<<" points"<<endl;

	while (!viewer.wasStopped ())
	{
		// range_image_borders_widget->spinOnce ();
		viewer.spinOnce ();
		pcl_sleep(0.01);
	}


}

/*
    @brief draw border to png
*/
void border2png()
{

}