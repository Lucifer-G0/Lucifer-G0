#pragma once
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef pcl::PointXYZ PointT;
class MyOBB
{
public:
    int OBB_no = 0;
    float width=0.0f;
    float height=0.0f;
    float depth=0.0f;

    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    MyOBB(pcl::PointCloud<PointT>::Ptr object_cloud);

    PointT min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    Eigen::Vector3f position;
    Eigen::Quaternionf quat;
};

MyOBB::MyOBB(pcl::PointCloud<PointT>::Ptr object_cloud)
{
    feature_extractor.setInputCloud(object_cloud);
    feature_extractor.compute();
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
    position = Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
    quat = Eigen::Quaternionf(rotational_matrix_OBB);
    width=max_point_OBB.x - min_point_OBB.x;
    height= max_point_OBB.y - min_point_OBB.y;
    depth= max_point_OBB.z - min_point_OBB.z;
}

