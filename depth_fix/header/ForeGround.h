#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZ PointT;

class ForeGround
{
public:
    ForeGround(pcl::PointCloud<PointT>::Ptr _cloud_foreground, float fore_seg_threshold_percent);
    pcl::PointCloud<PointT>::Ptr cloud_foreground;  //前景点云，随后变成前景点云的剩余点云
    void planar_seg();
    pcl::PointCloud<PointT>::Ptr extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster,int n=3);
    std::vector<cv::Point> extract_border_2D(pcl::PointCloud<PointT>::Ptr cloud_cluster,int n=3);
    void planar_repair();
    void object_detect();

private:
    float max_D=0.0f;    //最远平面距离，平面系数里的D
    float constant=570.3;
    int fore_seg_threshold; //水平面点数量阈值
    float hp_num;   //horizontal plane num, 水平面的数量

    std::vector<pcl::PointCloud<PointT>> plane_clouds;  //存储识别出来的独立水平面
    std::vector<pcl::ModelCoefficients> plane_coes;     //存储识别出的独立水平面的参数
};