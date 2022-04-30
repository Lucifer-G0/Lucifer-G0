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
    cv::Mat seg_image; //分割结果存储图像，默认为0,先填背景，再填前景平面，再填物体
    int width, height;                            //输入深度图像的宽度和高度,width=cols,height=rows
    DepthDetect(std::string depth_path = "00000-depth.png", int dimension = 2, float back_threshold_percent = 0.85f, float fore_seg_threshold_percent = 0.1f);

private:
    float constant = 570.3f; // RGBD数据集给定的相机参数，影响数据尺度
    int hp_start_no = 1, hp_no;          // horizontal plane num, 水平面的索引
    int object_start_no = 50, object_no; //检测出的物体索引
    int vp_start_no = 200, vp_no;        //背景面的索引
    int ground_no = 255;                 //地面的序号
    std::vector<float> object_points_nums;
    pcl::PointCloud<PointT>::Ptr cloud_junction;                 //交界点云，随后随部分物体加入背景点云

    // for BackGround
public:
    void back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients);
    void back_object_fill_2D(int point_enlarge=3);
    void back_plane_fix_2D_bak(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers);
    void back_cluster_extract(int dimension = 2, float back_ec_dis_threshold = 0.5f, float plane_seg_dis_threshold = 0.2f);
    void back_cluster_extract_2D(float back_ec_dis_threshold = 0.5f);
    cv::Mat get_Depth() { return Depth; }
    std::vector<cv::Rect> get_object_window();
    std::vector<cv::Rect> get_back_object_window();
    std::vector<cv::Rect> get_plane_window();

private:
    cv::Mat Depth;                                 //深度数据
    pcl::PointCloud<PointT>::Ptr cloud_background; //过滤得到的背景点云
    cv::Mat Mask;                                  //原始是否空洞点掩码，是空洞点为1，初始化时根据图像初始化，之后仅作为(多次填充)指示而不发生变化
    std::vector<float> min_depths;                 //维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准

    // for foreground
public:
    pcl::PointCloud<PointT>::Ptr cloud_foreground;                 //前景点云，随后变成前景点云的剩余点云
    std::vector<pcl::PointCloud<PointT>> plane_clouds;             //存储识别出来的独立水平面
    std::vector<pcl::PointCloud<PointT>> plane_border_clouds;      //存储识别出来的独立水平面的边缘
    std::vector<pcl::PointCloud<PointT>> plane_pure_border_clouds; //存储识别出来的独立水平面的纯净边缘，不包含因为遮挡造成的。
    pcl::PointCloud<PointT>::Ptr ground_cloud;                     //暂存/存储识别出来的地面,可能会出现更远的面成为地面

    void planar_seg(float plane_seg_dis_threshold = 0.13f, float layer_seg_dis_threshold = 0.3f);
    pcl::PointCloud<PointT>::Ptr extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n = 1);
    std::vector<cv::Point> extract_border_2D(int seg_no,int n = 1);
    std::vector<cv::Point> extract_border_2D_bak(pcl::PointCloud<PointT> cloud_cluster, int n = 1);
    
    void plane_fill_2D();

    void caculate_clean_border(bool fix = false);
    bool ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float fit_threshold_percent = 0.4f, float dis_threshold = 3.0f);
    int lines_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float line_threshold_percent = 0.2f, float line_dis_threshold = 0.05f, int plane_no = 999);
    void shape_fix(int plane_no);

    void object_detect();
    void object_detect_2D(float ec_dis_threshold = 0.25f, int follow_point_enlarge=0);

    void object_merge(float merge_threshold = 0.8f);
    void object_fill_2D();
    cv::Mat get_color_seg_image(cv::Mat &color_seg_image);

private:
    float max_D = 0.0f;                             //最远平面距离，平面系数里的D
    bool ground_is_stored = false;                  //表示ground_cloud内是否存有平面。
    int fore_seg_threshold;                         //水平面点数量阈值
    cv::Point2f get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p);
};