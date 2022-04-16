#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h> // Euclidean Cluster Extract
#include <pcl/console/time.h>

#include "NRLC.h"
#include "normal_estimation.h"
#include "transform.h"
#include "ObjectWindow.h"

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointCloud<PointT>::Ptr PointCloudTPtr;
typedef pcl::PointCloud<pcl::Normal> PointCloudN;
typedef pcl::PointCloud<pcl::Normal>::Ptr PointCloudNPtr;

PointCloudTPtr NRLC_Detect(PointCloudTPtr cloud_filtered, PointCloudNPtr normals, bool show = false);
PointCloudTPtr statisc_removal(PointCloudTPtr cloud_feature, bool test = false);
cv::Mat cluster_extract(cv::Mat image, PointCloudTPtr cloud, bool test = false);

int main(int argc, char **argv)
{
    PointCloudTPtr cloud(new PointCloudT);
    PointCloudTPtr cloud_filtered(new PointCloudT);
    PointCloudTPtr cloud_feature(new PointCloudT);
    PointCloudTPtr cloud_sor(new PointCloudT);
    PointCloudNPtr normals(new PointCloudN);

    cv::Mat image;
    
    std::string image_path = "../images/00000-color.png";
    std::string raw_pcd_path = "../raw_pcd/00000_cloud.pcd";

    //------------------load raw cloud pcd--------------------------------------------
    image = cv::imread(image_path, 1);

    pcl::console::print_highlight("Loading point cloud...\n");
    if (pcl::io::loadPCDFile<PointT>(raw_pcd_path, *cloud))
    {
        pcl::console::print_error("Error loading cloud file!\n");
        return 1;
    }
    //------------------fast normal estimation(KSearch)------------------------------
    cloud_filtered = passthrough_filter(cloud);
    normals = fast_normal_estimation(cloud_filtered);
    //------------------NRLC detect--------------------------------------------------
    pcl::console::TicToc tt;
    tt.tic();
    cloud_feature = NRLC_Detect(cloud_filtered, normals);
    std::cout<<"time spend "<<tt.toc()<<std::endl;
    //------------------stastic removal(移除部分数据异常导致的边界)-------------------------
    cloud_sor = statisc_removal(cloud_feature);
    //------------------Euclidean Cluster Extract----------------------------------------
    image = cluster_extract(image, cloud_sor);
    
    cv::imshow("return test", image);
    cv::waitKey();
    return 0;
}

cv::Mat cluster_extract(cv::Mat image, PointCloudTPtr cloud, bool test)
{
    pcl::PCDWriter writer;
    std::vector<ObjectWindow> object_windows;

    //  input check
    if (cloud->size() == 0)
    {
        std::cerr << "cluster_extract: input cloud has no data!" << std::endl;
    }
    else if (test)
    {
        std::cout << "border_points has " << cloud->size() << " points" << std::endl;
    }

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(1000000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

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
        image = object_window.draw(image);
        object_windows.push_back(object_window);

        if (test)
        {
            object_window.output();
            // cloud_cluster reset
            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            // save cluster to pcd
            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size() << " data points." << std::endl;
            std::stringstream ss;
            ss << "../test_output/clusters/cloud_cluster_" << j << ".pcd";
            writer.write<PointT>(ss.str(), *cloud_cluster, false);
        }
        j++;
    }

    if (test)
    {
        cv::imshow("test show", image);
        cv::waitKey();
    }

    return image;
}

PointCloudTPtr statisc_removal(PointCloudTPtr cloud_feature, bool test)
{
    PointCloudTPtr cloud_sor(new PointCloudT);

    if (test)
    {
        std::cerr << "Cloud before filtering: " << cloud_feature->size() << std::endl;
    }

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_feature);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_sor);

    if (test)
    {
        std::cerr << "Cloud after filtering: " << cloud_sor->size() << std::endl;

        pcl::PCDWriter writer;
        writer.write<pcl::PointXYZ>("../test_output/sor/inliers.pcd", *cloud_sor, false);

        sor.setNegative(true);
        sor.filter(*cloud_sor);
        writer.write<pcl::PointXYZ>("../test_output/sor/outliers.pcd", *cloud_sor, false);

        //还原cloud_sor为感兴趣点
        sor.setNegative(false);
        sor.filter(*cloud_sor);
    }

    return cloud_sor;
}

PointCloudTPtr NRLC_Detect(PointCloudTPtr cloud_filtered, PointCloudNPtr normals, bool show)
{
    NRLC nrlc;
    nrlc.setInputCloud(cloud_filtered);
    nrlc.setNormals(normals);
    nrlc.setParams(100, 60, 0.6, 0.5);
    std::vector<int> vec_n_feature;
    nrlc.detect(vec_n_feature);
    nrlc.EDE(vec_n_feature, 10, 3, 1);
    nrlc.refine(3, vec_n_feature);

    PointCloudTPtr feature_cloud(new PointCloudT);
#pragma omp parallel
#pragma omp for
    for (size_t i = 0; i < cloud_filtered->size(); i++)
    {
        PointT point;
        point.x = cloud_filtered->points[i].x;
        point.y = cloud_filtered->points[i].y;
        point.z = cloud_filtered->points[i].z;
        if (vec_n_feature[i] == 1 || vec_n_feature[i] == 2 || vec_n_feature[i] == 3)
        {
            feature_cloud->push_back(point);
        }
    }

    if (show)
    {
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
            // if (vec_n_feature[i] == 3) // border
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
        pcl::io::savePCDFile("nrlc_test_out_max_40_000.pcd", *visual_cloud);
        std::cout << "save finish" << std::endl;

        pcl::visualization::PCLVisualizer visualizer("show result");
        visualizer.addPointCloud(visual_cloud);
        visualizer.spin();
        while (!visualizer.wasStopped())
        {
            visualizer.spinOnce(100);
            pcl_sleep(0.01);
        }
    }

    return feature_cloud;
}