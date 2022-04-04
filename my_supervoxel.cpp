#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/supervoxel_clustering.h>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

#include <stdlib.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

// Types
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;


int
main (int argc, char ** argv)
{
  pcl::PCDWriter writer;

  if (argc < 2)
  {
    pcl::console::print_error ("Syntax is: %s <pcd-file> \n "
                                "--NT Dsables the single cloud transform \n"
                                "-v <voxel resolution>\n-s <seed resolution>\n"
                                "-c <color weight> \n-z <spatial weight> \n"
                                "-n <normal_weight>\n", argv[0]);
    return (1);
  }


  PointCloudT::Ptr cloud1 (new PointCloudT);
  PointCloudT::Ptr cloud (new PointCloudT);
  pcl::console::print_highlight ("Loading point cloud...\n");
  if (pcl::io::loadPCDFile<PointT> (argv[1], *cloud1))
  {
    pcl::console::print_error ("Error loading cloud file!\n");
    return (1);
  }

  float rot = 0.5f;
  if (pcl::console::find_switch (argc, argv, "-rot"))
    pcl::console::parse (argc, argv, "-rot", rot);

  // remove StatisticalOutlier
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud (cloud1);
  sor.setMeanK (50);
  sor.setStddevMulThresh (rot);
  sor.setNegative(true);
 
  pcl::PointCloud<PointT>::Ptr cloud_remove (new pcl::PointCloud<PointT> ());
  // Remove the planar inliers, extract the rest
  sor.filter (*cloud_remove);
  
  std::cerr << "PointCloud representing the remove component: " << cloud_remove->size () << " data points." << std::endl;
  writer.write ("remove.pcd", *cloud_remove, false);

  sor.setNegative(false);
  sor.filter(*cloud);


  bool disable_transform = pcl::console::find_switch (argc, argv, "--NT");

  float voxel_resolution = 0.008f;
  bool voxel_res_specified = pcl::console::find_switch (argc, argv, "-v");
  if (voxel_res_specified)
    pcl::console::parse (argc, argv, "-v", voxel_resolution);

  float seed_resolution = 0.1f;
  bool seed_res_specified = pcl::console::find_switch (argc, argv, "-s");
  if (seed_res_specified)
    pcl::console::parse (argc, argv, "-s", seed_resolution);

  float color_importance = 0.2f;
  if (pcl::console::find_switch (argc, argv, "-c"))
    pcl::console::parse (argc, argv, "-c", color_importance);

  float spatial_importance = 0.4f;
  if (pcl::console::find_switch (argc, argv, "-z"))
    pcl::console::parse (argc, argv, "-z", spatial_importance);

  float normal_importance = 1.0f;
  if (pcl::console::find_switch (argc, argv, "-n"))
    pcl::console::parse (argc, argv, "-n", normal_importance);

  float ct = 0.1f;
  if (pcl::console::find_switch (argc, argv, "-ct"))
    pcl::console::parse (argc, argv, "-ct", ct);
  
  

  //////////////////////////////  //////////////////////////////
  ////// This is how to use supervoxels
  //////////////////////////////  //////////////////////////////

  pcl::SupervoxelClustering<PointT> super (voxel_resolution, seed_resolution);
  if (disable_transform)
    super.setUseSingleCameraTransform (false);
  super.setInputCloud (cloud);
  super.setColorImportance (color_importance);
  super.setSpatialImportance (spatial_importance);
  super.setNormalImportance (normal_importance);

  std::map <std::uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

  pcl::console::print_highlight ("Extracting supervoxels!\n");
  super.extract (supervoxel_clusters);

  int num=supervoxel_clusters.size ();
  pcl::console::print_info ("Found %d supervoxels\n", num);

  PointCloudT::Ptr super_voxel_centroid_cloud(new pcl::PointCloud<PointT>);
  for(auto supervoxel_cluster:supervoxel_clusters)
  {
    super_voxel_centroid_cloud->push_back(supervoxel_cluster.second->centroid_);
  }


  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<PointT> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.03);

  seg.setInputCloud (super_voxel_centroid_cloud);
  seg.segment (*inliers, *coefficients);

  pcl::ExtractIndices<PointT> extract;
  // Extract the planar inliers from the input cloud
  extract.setInputCloud (super_voxel_centroid_cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  

  extract.filter (*cloud_plane);
  std::cerr << "PointCloud representing the planar component: " << cloud_plane->size () << " data points." << std::endl;
  writer.write ("plane.pcd", *cloud_plane, false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_object (new pcl::PointCloud<PointT> ());
  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_object);
  
  std::cerr << "PointCloud representing the planar component: " << cloud_object->size () << " data points." << std::endl;
  writer.write ("object.pcd", *cloud_object, false);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
  tree->setInputCloud (cloud_object);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance (ct); // 20cm -s 0.1
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_object);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
    for (const auto& idx : it->indices)
    {
        cloud_cluster->push_back ((*cloud_object)[idx]); //*
        cout << idx << endl;
    }
      
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    PointT min,max;
    pcl::getMinMax3D(*cloud_cluster,min,max);
    cout << "->min_x = " << min.x << endl;
	  cout << "->min_y = " << min.y << endl;
	  cout << "->min_z = " << min.z << endl;
	  cout << "->max_x = " << max.x << endl;
	  cout << "->max_y = " << max.y << endl;
	  cout << "->max_z = " << max.z << endl;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<PointT> (ss.str (), *cloud_cluster, false); //*
    j++;
  }
  std::stringstream ss;
  ss<<"pcl_viewer -cam points"<<argv[1][6]<<".cam ";

  for(int i=0;i<j;i++)
  {
    
    ss << " -ps 20  cloud_cluster_" << i << ".pcd";
  }
  ss<<" -ps 1 "<<argv[1];
  std::cerr<<ss.str()<<std::endl;

  system(ss.str().c_str());

}

