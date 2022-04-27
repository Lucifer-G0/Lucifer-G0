#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZ PointT;

class ForeGround
{
public:
    pcl::PointCloud<PointT>::Ptr cloud_foreground; //前景点云，随后变成前景点云的剩余点云
    std::vector<pcl::PointCloud<PointT>> plane_clouds; //存储识别出来的独立水平面
    std::vector<pcl::PointCloud<PointT>> plane_border_clouds; //存储识别出来的独立水平面的边缘
    std::vector<pcl::PointCloud<PointT>> plane_pure_border_clouds; //存储识别出来的独立水平面的纯净边缘，不包含因为遮挡造成的。
    cv::Mat seg_image=cv::Mat::zeros(480,640,CV_8U);    //分割结果存储图像，无物体为0，用不同的标号表示不同物体。0~49:hp;50~99:vp;100~ object
    
    ForeGround(pcl::PointCloud<PointT>::Ptr _cloud_foreground, float fore_seg_threshold_percent);
    
    void planar_seg();
    pcl::PointCloud<PointT>::Ptr extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n = 3);
    std::vector<cv::Point> extract_border_2D(pcl::PointCloud<PointT> cloud_cluster, int n = 3);

    //拟合函数，用于提取纯净边界点，剔除因遮挡或缺失造成的边界

    bool ellipse_fit( pcl::PointCloud<PointT>::Ptr border_cloud);
    void lines_fit( pcl::PointCloud<PointT>::Ptr border_cloud, int plane_no=999);

    void border_clean();
    void object_detect_2D_bak();
    void object_detect_2D();
    

    void shape_fit();
    

private:
    int hp_start_no=1;
    int object_start_no=50;

    float max_D = 0.0f; //最远平面距离，平面系数里的D
    float constant = 570.3;
    int hp_no;           // horizontal plane num, 水平面的索引
    int object_no;   //检测出的物体索引
    int fore_seg_threshold; //水平面点数量阈值

    std::vector<pcl::ModelCoefficients> plane_coes; //存储识别出的独立水平面的参数
    cv::Point2f get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p);
    
};