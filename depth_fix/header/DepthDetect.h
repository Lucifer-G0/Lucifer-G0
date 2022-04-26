#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZ PointT;

class DepthDetect
{
// communal
public:
    cv::Mat seg_image=cv::Mat::zeros(480,640,CV_8U);    //分割结果存储图像，默认为0,先填背景，再填前景平面，再填物体
    int width;
	int height;
    DepthDetect(int dimension=2);
private:
    float constant=570.3;   //RGBD数据集给定的相机参数，影响数据尺度
// for BackGround
public:
	cv::Mat get_Depth(){return Depth;}
    void back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers,pcl::ModelCoefficients::Ptr coefficients);
	void back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers);
	void back_cluster_extract(int dimension=2);
private:
    cv::Mat Depth;  //深度数据
    pcl::PointCloud<PointT>::Ptr cloud_background;//过滤得到的背景点云
	int vp_start_no=200;
	int vp_no;       //背景面的索引
	cv::Mat Mask;	//原始是否空洞点掩码，是空洞点为1，初始化时根据图像初始化，之后仅作为(多次填充)指示而不发生变化
	std::vector<float> min_depths;	//维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准
    

//for foreground
public:
    pcl::PointCloud<PointT>::Ptr cloud_foreground; //前景点云，随后变成前景点云的剩余点云
    std::vector<pcl::PointCloud<PointT>> plane_clouds; //存储识别出来的独立水平面
    std::vector<pcl::PointCloud<PointT>> plane_border_clouds; //存储识别出来的独立水平面的边缘
    std::vector<pcl::PointCloud<PointT>> plane_pure_border_clouds; //存储识别出来的独立水平面的纯净边缘，不包含因为遮挡造成的。
    
    void planar_seg();
    pcl::PointCloud<PointT>::Ptr extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n = 3);
    std::vector<cv::Point> extract_border_2D(pcl::PointCloud<PointT> cloud_cluster, int n = 3);

    void border_clean();
    bool ellipse_fit( pcl::PointCloud<PointT>::Ptr border_cloud);
    void lines_fit( pcl::PointCloud<PointT>::Ptr border_cloud, int plane_no=999);

    void object_detect_2D_bak();
    void object_detect_2D();
    void shape_fit();
private:
    int hp_start_no=100;
    int object_start_no=50;

    float max_D = 0.0f; //最远平面距离，平面系数里的D
    int hp_no;           // horizontal plane num, 水平面的索引
    int object_no;   //检测出的物体索引
    int fore_seg_threshold; //水平面点数量阈值

    std::vector<pcl::ModelCoefficients> plane_coes; //存储识别出的独立水平面的参数
    cv::Point2f get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p);
};