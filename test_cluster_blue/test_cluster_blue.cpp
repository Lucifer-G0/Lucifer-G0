#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/segmentation/extract_clusters.h> // Euclidean Cluster Extract
#include <string>
#include "ObjectWindow.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

int main(int argc, char **argv)
{
    PointCloudT::Ptr cloud(new PointCloudT);
    pcl::PCDWriter writer;
    std::string no=argv[1];
    std::string image_path = "../imgs/000"+no+"-color.png";
    std::string pcd_path = "../cloud/000"+no+"_cloud_blue.pcd";

    // load cloud from pcd file
    pcl::console::print_highlight("Loading point cloud...\n");
    if (pcl::io::loadPCDFile<PointT>(pcd_path, *cloud))
    {
        pcl::console::print_error("Error loading cloud file!\n");
        return (1);
    }
    // --------------------------------
    // Euclidean Cluster Extract
    // -----------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);
    std::cout << "border_points has " << cloud->size() << " points" << std::endl;
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.2);
    ec.setMinClusterSize(200);
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    std::vector<ObjectWindow> object_windows;
    cv::Mat image;
    image = cv::imread(image_path, 1);

    int j = 0;
    // 外循环 循环所有聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {

        ObjectWindow object_window;

        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        //内循环 循环一个聚类内部的所有点
        for (const auto &idx : it->indices)
        {
            PointT border_point = (*cloud)[idx];
            cloud_cluster->push_back(border_point); //*
            object_window.add_point(border_point);
        }

        object_window.update();
        object_window.output();
        image = object_window.draw(image);
        object_windows.push_back(object_window);

        // cloud_cluster reset
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        // save cluster to pcd
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
        std::stringstream ss;
        ss << "cloud_cluster_" << j << ".pcd";
        writer.write<PointT>(ss.str(), *cloud_cluster, false); //*

        j++;
    }

    cv::imshow("show",image);
    while(true)
    {
        if(cv::waitKey(100)==27)
            break;
    }

    // std::stringstream ss;
    // // ss << "pcl_viewer -cam points" << argv[1][6] << ".cam ";
    // ss << "pcl_viewer ";
    // for (int i = 0; i < j; i++)
    // {

    //     ss << " -ps 5  cloud_cluster_" << i << ".pcd";
    // }
    // ss << " -ps 1 " << pcd_path;
    // std::cerr << ss.str() << std::endl;

    // int a = system(ss.str().c_str());

    return 0;
}

// void show()
// {
//     std::vector<pcl::PointIndices> cluster_indices;
//       int j = 0;
//   // 外循环 循环所有聚类
//   for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
//   {

//     ObjectWindow object_window;

//     pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
//     //内循环 循环一个聚类内部的所有点
//     for (const auto& idx : it->indices)
//     {
//         PointT border_point=(*border_points_ptr)[idx];
//         cloud_cluster->push_back (border_point); //*
//         object_window.add_point(border_point);
//     }

//     object_window.update();
//     object_window.output();
//     image=object_window.draw(image);
//     object_windows.push_back(object_window);

//     // cloud_cluster reset
//     cloud_cluster->width = cloud_cluster->size ();
//     cloud_cluster->height = 1;
//     cloud_cluster->is_dense = true;
//     // save cluster to pcd
//     std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
//     std::stringstream ss;
//     ss << "cloud_cluster_" << j << ".pcd";
//     writer.write<PointT> (ss.str (), *cloud_cluster, false); //*

//     j++;
//   }
// }