#pragma once
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointXYZRGB PointC;

class DepthDetect
{
    // communal
public:
    cv::Mat seg_image; //分割结果存储图像，默认为0,先填背景，再填前景平面，再填物体
    int width, height; //输入深度图像的宽度和高度,width=cols,height=rows
    int dimension;
    DepthDetect(std::string depth_path = "00000-depth.png", int _dimension = 2, float back_threshold_percent = 0.85f, float fore_seg_threshold_percent = 0.1f);

private:
    float constant = 570.3f;                     // RGBD数据集给定的相机参数，影响数据尺度
    int hp_start_no = 1, hp_no;                  // horizontal plane num, 水平面的索引
    int object_start_no = 50, object_no;         //检测出的物体索引
    int vp_start_no = 200, vp_no;                //背景面的索引
    int ground_no = 255;                         //地面的序号
    int OBB_no = 0;                              //包围盒数量,显示id使用
    pcl::PointCloud<PointT>::Ptr cloud_junction; //交界点云，随后随部分物体加入背景点云

    // for BackGround
public:
    void back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients, int back_object_no);
    void back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers);
    void back_object_fill_2D(int point_enlarge = 3);

    void back_cluster_extract(float back_ec_dis_threshold = 0.5f, bool fix = false, float plane_seg_dis_threshold = 0.2f);
    void back_cluster_extract_2D(float back_ec_dis_threshold = 0.5f);

    std::vector<cv::Rect> get_object_window();
    std::vector<cv::Rect> get_back_object_window();
    std::vector<cv::Rect> get_plane_window();
    cv::Mat get_color_seg_image(cv::Mat &color_seg_image, int point_enlarge = 0);

    cv::Mat get_Depth() { return Depth; }
    void show_3D(bool box = false);
    void add_detect_OBB(pcl::PointCloud<PointT>::Ptr object_cloud, pcl::visualization::PCLVisualizer::Ptr viewer, cv::Vec3f color);
    pcl::PointCloud<PointC>::Ptr get_color_pointcloud();

private:
    cv::Mat Depth;                                 //深度数据
    pcl::PointCloud<PointT>::Ptr cloud_background; //过滤得到的背景点云
    cv::Mat Mask;                                  //原始是否空洞点掩码，是空洞点为1，初始化时根据图像初始化，之后仅作为(多次填充)指示而不发生变化
    std::vector<float> min_depths;                 //维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准

    // for foreground
public:
    // pcl::PointCloud<PointT>::Ptr cloud_primitive;                 //前景点云，随后变成前景点云的剩余点云
    pcl::PointCloud<PointT>::Ptr cloud_foreground;                 //前景点云，随后变成前景点云的剩余点云
    std::vector<pcl::PointCloud<PointT>> plane_clouds;             //存储识别出来的独立水平面
    std::vector<pcl::PointCloud<PointT>> plane_border_clouds;      //存储识别出来的独立水平面的边缘
    std::vector<pcl::PointCloud<PointT>> plane_pure_border_clouds; //存储识别出来的独立水平面的纯净边缘，不包含因为遮挡造成的。
    pcl::PointCloud<PointT>::Ptr ground_cloud;                     //暂存/存储识别出来的地面,可能会出现更远的面成为地面
    std::vector<pcl::PointCloud<PointT>> object_clouds;            //存储识别出来的独立前景物体
    std::vector<pcl::PointCloud<PointT>> back_object_clouds;       //存储识别出来的独立背景物体
    pcl::ModelCoefficients ground_coes;                            //地面的模型系数
    float plane_seg_dis_threshold;                                 //由planar_seg指定,存储给plane_complete备用
    float layer_seg_dis_threshold;

    void planar_seg(float _plane_seg_dis_threshold = 0.13f, float layer_seg_dis_threshold = 0.3f);
    bool plane_check(pcl::PointCloud<PointT>::Ptr cloud_cluster, float is_line_threshold = 0.8f, float line_dis_threshold = 0.13f);
    void ground_complete();
    pcl::PointCloud<PointT>::Ptr extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n = 1);
    std::vector<cv::Point> extract_border_2D(int seg_no, int n = 1);
    std::vector<cv::Point> extract_border_2D_bak(pcl::PointCloud<PointT> cloud_cluster, int n = 1);

    void plane_fill_2D();

    void caculate_clean_border(bool fix = false);
    bool ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float fit_threshold_percent = 0.4f, float dis_threshold = 3.0f, int plane_no = 999);
    int lines_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float line_threshold_percent = 0.2f, float line_dis_threshold = 0.05f, int plane_no = 999);
    void shape_fix(int plane_no);

    void object_detect(float ec_dis_threshold = 0.25f);
    void object_detect_2D(float ec_dis_threshold = 0.25f, int follow_point_enlarge = 0);

    void object_merge(float merge_threshold = 0.8f);
    void object_merge_2D(float merge_threshold = 0.8f);
    void object_fill_2D();

private:
    float max_D = 0.0f;            //最远平面距离，平面系数里的D
    bool ground_is_stored = false; //表示ground_cloud内是否存有平面。
    int fore_seg_threshold;        //水平面点数量阈值
    cv::Point2f get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p);
};