#include <opencv2/opencv.hpp>
#include <math.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波器头文件
#include <pcl/filters/random_sample.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/model_outlier_removal.h>

#include "DepthDetect.h"
#include "ObjectWindow.h"
#include "transform.h"
#include "MyColor.hpp"

/*
    @param  dimension: (default=2),2或3,表示期待结果为2维或者3维
    @param  back_threshold_percent: (default= 0.85f) 用于计算背景的深度阈值，百分比形式。0.85比较合适？
    @param  fore_seg_threshold_percent: (default=0.1f)前景分割是否平面阈值，前景点云大小的百分比
*/
DepthDetect::DepthDetect(std::string depth_path, int dimension, float back_threshold_percent, float fore_seg_threshold_percent)
{
    float back_threshold = 0.0f;
    float max_depth = 50.0f;

    Depth = cv::imread(depth_path, -1);
    Depth.convertTo(Depth, CV_32F);
    width = Depth.cols;
    height = Depth.rows;

    seg_image = cv::Mat::zeros(height, width, CV_8U);

    Mask = cv::Mat::zeros(height, width, CV_8U);
    for (int r = 0; r < Depth.rows; r++)
        for (int c = 0; c < Depth.cols; c++)
            if (Depth.at<float>(r, c) == 0) //是空洞点
                Mask.at<uchar>(r, c) = 1;

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    cloud = depth2cloud(depth_path);
    // pcl::io::savePCDFile("before.pcd", *cloud);
    // std::cout << "before: " << cloud->size() << std::endl;

    // pcl::RandomSample<PointT> rs;
    // rs.setInputCloud(cloud);
    // //设置输出点的数量
    // rs.setSample(cloud->size() * 0.1f);
    // //下采样并输出到cloud_out
    // rs.filter(*cloud);

    // pcl::io::savePCDFile("after.pcd", *cloud);
    // std::cout << "after: " << cloud->size() << std::endl;

    //--------------计算背景的深度阈值------------------------------------------
    std::vector<float> sorted_Depth;
    for (auto &point : *cloud)
    {
        //计算阈值应该去除零点？
        if (point.z != 0)
        {
            sorted_Depth.push_back(point.z);
        }
    }
    std::sort(sorted_Depth.begin(), sorted_Depth.end());
    back_threshold = sorted_Depth[(int)(sorted_Depth.size() * back_threshold_percent)]; //根据百分比计算得到阈值
    max_depth = sorted_Depth[sorted_Depth.size() - 1] + 0.001;                          //获得最大值，不清楚过滤的开闭，因而加一点避免最大值被过滤

    //---------根据阈值过滤出背景,分割出背景和前景------------------------------------
    cloud_background = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    cloud_foreground = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    cloud_junction = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");

    pass.setFilterLimits(back_threshold + 0.5, max_depth); //阈值到最大值
    pass.filter(*cloud_background);                        //背景点云
    pass.setFilterLimits(back_threshold - 0.5, back_threshold + 0.5 - 0.001);
    pass.filter(*cloud_junction); //交界点云
    pass.setFilterLimits(0.001, back_threshold - 0.5 - 0.001);
    pass.filter(*cloud_foreground); //前景点云,需注意前景必须去除零点，因为零点占相当大部分
    // pcl::io::savePCDFile("foreground.pcd", *cloud_foreground);
    // std::cout << "foreground: " << cloud_foreground->size() << std::endl;
    // pcl::io::savePCDFile("background.pcd", *cloud_background);
    // std::cout << "background: " << cloud_background->size() << std::endl;
    // pcl::io::savePCDFile("junction.pcd", *cloud_junction);
    // std::cout << "junction: " << cloud_junction->size() << std::endl;

    fore_seg_threshold = fore_seg_threshold_percent * cloud_foreground->size();

    hp_no = hp_start_no;
    object_no = object_start_no;
    vp_no = vp_start_no;
}

/*
    根据seg_image中的各种标号，制作彩色化分割结果，背景为灰色系，前景平面为紫色系，空洞为黑色，地面为紫色，其他为物体
    [IN] seg_image
    @return cv::Mat color_seg_image(height, width, CV_8UC3),彩色图
*/
cv::Mat DepthDetect::get_color_seg_image(cv::Mat &color_seg_image)
{
    MyColor my_color;
    int rows = height, cols = width;

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int seg_no = seg_image.at<uchar>(r, c);
            if (seg_no == 0)
            {
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.hole_color;
            }
            else if (seg_no == 255)
            {
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.ground_color;
            }
            else if (seg_no >= vp_start_no)
            {
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.back_colors[((seg_no - vp_start_no) * 3) % my_color.bc_size];
            }
            else if (seg_no >= object_start_no)
            {
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.object_colors[((seg_no - object_start_no) * 8) % my_color.oc_size]; //对序号做变换实现相邻序号物体较大颜色跨度。
            }
            else if (seg_no >= hp_start_no)
            {
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.plane_colors[(seg_no - hp_start_no) % my_color.pc_size];
            }
        }
    }
    return color_seg_image;
}

//测试环境：ubuntu显示效果左手坐标系，要对z取反才正常显示。
void DepthDetect::show_3D()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    //------------------将物体点云彩色化并加入----------------------------------------------
    MyColor myColor;
    int object_num = object_clouds.size();
    for (int i = 0; i < object_num; i++)
    {
        pcl::PointCloud<PointT>::Ptr object_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        for (PointT point : object_clouds[i])
        {
            object_cloud->push_back(PointT(point.x, point.y, -point.z));
        }
        cv::Vec3b color = myColor.object_colors[(i * 7) % myColor.oc_size];
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(object_cloud, color[0], color[1], color[2]);
        viewer->addPointCloud<pcl::PointXYZ>(object_cloud, single_color, "object_" + std::to_string(i), 0);
        add_detect_OBB(object_cloud, viewer, cv::Vec3f(0, 0, 1));
    }
    int back_object_num = back_object_clouds.size();
    for (int i = 0; i < back_object_num; i++)
    {
        pcl::PointCloud<PointT>::Ptr back_object_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        for (PointT point : back_object_clouds[i])
        {
            back_object_cloud->push_back(PointT(point.x, point.y, -point.z));
        }
        cv::Vec3b color = myColor.back_colors[(i * 3) % myColor.oc_size];
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(back_object_cloud, color[0], color[1], color[2]);
        viewer->addPointCloud<pcl::PointXYZ>(back_object_cloud, single_color, "back_object_" + std::to_string(i), 0);
        add_detect_OBB(back_object_cloud, viewer, cv::Vec3f(0, 1, 0));
    }
    int plane_num = plane_clouds.size();
    for (int i = 0; i < plane_num; i++)
    {
        pcl::PointCloud<PointT>::Ptr plane_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        for (PointT point : plane_clouds[i])
        {
            plane_cloud->push_back(PointT(point.x, point.y, -point.z));
        }
        cv::Vec3b color = myColor.plane_colors[i % myColor.pc_size];
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(plane_cloud, color[0], color[1], color[2]);
        viewer->addPointCloud<pcl::PointXYZ>(plane_cloud, single_color, "plane_" + std::to_string(i), 0);
        add_detect_OBB(plane_cloud, viewer, cv::Vec3f(1, 0, 0));
    }
    pcl::PointCloud<PointT>::Ptr view_ground_cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
    for (PointT point : *ground_cloud)
    {
        view_ground_cloud->push_back(PointT(point.x, point.y, -point.z));
    }
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(view_ground_cloud, myColor.ground_color[0], myColor.ground_color[1], myColor.ground_color[2]);
    viewer->addPointCloud<pcl::PointXYZ>(view_ground_cloud, single_color, "ground", 0);
    add_detect_OBB(view_ground_cloud, viewer, cv::Vec3f(0.5, 0, 0.5));

    viewer->initCameraParameters();
    viewer->setCameraPosition(-1, -2, 5, -1, 0, 0, 0);
    viewer->setRepresentationToWireframeForAllActors(); //显示为线框

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        pcl_sleep(0.1);
    }
}

void DepthDetect::add_detect_OBB(pcl::PointCloud<PointT>::Ptr object_cloud, pcl::visualization::PCLVisualizer::Ptr viewer, cv::Vec3f color)
{
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(object_cloud);
    feature_extractor.compute();

    PointT min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    Eigen::Quaternionf quat(rotational_matrix_OBB);
    viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB_" + std::to_string(OBB_no));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color[0], color[1], color[2], "OBB_" + std::to_string(OBB_no));
    viewer->setRepresentationToWireframeForAllActors(); //显示为线框

    OBB_no++;
}

/*
    根据分割(合并)结果,将不同物体赋予色彩组合成为彩色点云.
    [IN] object_clouds, back_object_clouds, plane_clouds, ground_cloud
    @return 分割结果点云
*/
pcl::PointCloud<PointC>::Ptr DepthDetect::get_color_pointcloud()
{
    pcl::PointCloud<PointC>::Ptr color_pointcloud = pcl::PointCloud<PointC>::Ptr(new pcl::PointCloud<PointC>);
    MyColor myColor;
    int object_num = object_clouds.size();
    for (int i = 0; i < object_num; i++)
    {
        cv::Vec3b color = myColor.object_colors[(i * 8) % myColor.oc_size];
        for (PointT point : object_clouds[i])
        {
            color_pointcloud->push_back(PointC(point.x, point.y, -point.z, color[0], color[1], color[2]));
        }
    }
    int back_object_num = back_object_clouds.size();
    for (int i = 0; i < back_object_num; i++)
    {
        cv::Vec3b color = myColor.back_colors[(i * 3) % myColor.oc_size];
        for (PointT point : back_object_clouds[i])
        {
            color_pointcloud->push_back(PointC(point.x, point.y, -point.z, color[0], color[1], color[2]));
        }
    }
    int plane_num = plane_clouds.size();
    for (int i = 0; i < plane_num; i++)
    {
        cv::Vec3b color = myColor.plane_colors[i % myColor.pc_size];
        for (PointT point : plane_clouds[i])
        {
            color_pointcloud->push_back(PointC(point.x, point.y, -point.z, color[0], color[1], color[2]));
        }
    }
    cv::Vec3b color = myColor.ground_color;
    for (PointT point : *ground_cloud)
    {
        color_pointcloud->push_back(PointC(point.x, point.y, -point.z, color[0], color[1], color[2]));
    }
    return color_pointcloud;
}

/*
    矩形修复，背景数据精度较低，比较杂乱，缺失较大，但总体上其位置分布相对集中，先聚类，然后采用较大的阈值在每个聚类内拟合出一个平面。
    [IN] cloud_background: 分割出的背景点云数据。
    [OUT]: Depth 每个聚类拟合一个矩形平面，将背景的空洞修复值填充到Depth | seg_image 根据所在平面序号填充 seg_image
    @param dimension: 维度(2 or 3) 2维在seg_image矩形填充平面序号，3维对深度数据Depth进行矩形区域恢复
    @param back_ec_dis_threshold: (default=0.5) 背景欧几里德聚类的距离阈值，背景数据精度低，相对杂乱，需要更大阈值，但不好控制
    @param plane_seg_dis_threshold: (default=0.2) 平面分割距离阈值
*/
void DepthDetect::back_cluster_extract(float back_ec_dis_threshold, bool fix, float plane_seg_dis_threshold)
{
    //----------------------对背景聚类-------------------------------------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_background);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(back_ec_dis_threshold);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(cloud_background->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_background);
    ec.extract(cluster_indices);

    //--------------------遍历聚类，提取背景物体点云,如果修复:每个聚类中找出一个平面，用平面对矩形区域作修复-------------------------
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointT>::Ptr back_obj_cloud_cluster(new pcl::PointCloud<PointT>);
        for (const auto &idx : it->indices)
            back_obj_cloud_cluster->push_back((*cloud_background)[idx]);
        back_obj_cloud_cluster->width = back_obj_cloud_cluster->size();
        back_obj_cloud_cluster->height = 1;
        back_obj_cloud_cluster->is_dense = true;
        back_object_clouds.push_back(*back_obj_cloud_cluster);
        if (fix)
        {
            //------------------- Create the segmentation object for the planar model and set all the parameters----------------
            pcl::SACSegmentation<PointT> seg;
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setMaxIterations(100);
            seg.setDistanceThreshold(plane_seg_dis_threshold);
            // -- --Segment the largest planar component from the remaining cloud------- -- -- -- -
            seg.setInputCloud(back_obj_cloud_cluster);
            seg.segment(*inliers, *coefficients);
            back_plane_fix(back_obj_cloud_cluster, inliers, coefficients, vp_no);
        }
        //一个背景平面结束，背景平面序号加一
        vp_no++;
    }
}

/*
    对一个背景平面进行深度填充(点云层面、原深度图层面)
    [IN] 欧几里德聚类分割出的背景平面
    [OUT] Depth 根据所在平面系数将其矩形区域修复填充
    [OUT] back_object_clouds 将点云存入对应背景物体点云
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
    @param inliers: 聚类中平面内点的索引，根据内点计算出矩形区域
    @param coefficients: 该聚类内平面的系数，用于进行深度数据修复，在拟合平面上进行填充。
*/
void DepthDetect::back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients, int back_object_no)
{
    ObjectWindow object_window;
    float A, B, C, D;
    A = coefficients->values[0];
    B = coefficients->values[1];
    C = coefficients->values[2];
    D = coefficients->values[3];

    //从聚类中提取出平面上的点,计算窗口
    for (const auto &idx : inliers->indices)
    {
        PointT border_point = (*cloud_cluster)[idx];
        object_window.add_point(border_point);
    }

    object_window.update();

    //遍历区域，将所有矩形区域内深度修复,平面方程Ax+By+Cz+D=0->z
    //行遍历,即使并行效率也没有明显提高，该部分应该耗时不大
    for (int r = object_window.topleft_x; r < object_window.topleft_x + object_window.height; r++)
    {
        //列遍历
        for (int c = object_window.topleft_y; c < object_window.topleft_y + object_window.width; c++)
        {
            if (Mask.at<uchar>(r, c) == 1 && Depth.at<float>(r, c) == 0) //是空洞点
            {
                //------------根据模型计算它的值
                float z = -D * constant * 1000. / (A * r + B * c + C * constant);
                Depth.at<float>(r, c) = z;
                float x = r * z / constant / 1000.;
                float y = c * z / constant / 1000.;
                back_object_clouds[back_object_no - vp_start_no].push_back(PointT(x, y, z));
            }
            else if (Mask.at<uchar>(r, c) == 1) //已经被填充过的空洞点,优先填充为深的
            {
                float z = -D * constant * 1000. / (A * r + B * c + C * constant);
                if (z > Depth.at<float>(r, c))
                {
                    Depth.at<float>(r, c) = z;
                    float x = r * z / constant / 1000.;
                    float y = c * z / constant / 1000.;
                    back_object_clouds[back_object_no - vp_start_no].push_back(PointT(x, y, z));
                }
            }
        }
    }
}

/*
    背景数据精度较低，比较杂乱，缺失较大，但总体上其位置分布相对集中，先聚类，然后赋值。修复的时间代价较大，聚类不好控制。
    [IN] cloud_background: 分割出的背景点云数据。
    [OUT]: seg_image 根据所在平面序号填充 seg_image
    @param back_ec_dis_threshold: (default=0.5) 背景欧几里德聚类的距离阈值，背景数据精度低，相对杂乱，需要更大阈值，但不好控制
*/
void DepthDetect::back_cluster_extract_2D(float back_ec_dis_threshold)
{
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    //----------------------对背景聚类---------------------------------------------------------------也可以考虑先下采样再聚类？
    tree->setInputCloud(cloud_background);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(back_ec_dis_threshold);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(cloud_background->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_background);
    ec.extract(cluster_indices);

    //--------------------遍历聚类，每个聚类中找出一个平面，并用平面对矩形区域作修复---------------------------------
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        for (const auto &idx : it->indices)
            cloud_cluster->push_back((*cloud_background)[idx]);
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        //遍历,某高度平面内的一个独立平面的内点，不存在写冲突，即使已被写过，前景的优先级更高
        for (auto &point : *cloud_cluster)
        {
            int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
            int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
            seg_image.at<uchar>(r, c) = vp_no;
        }
        vp_no++; //一个聚类结束，背景平面序号加一
    }
}

/*
    根据背景物体已经画在seg_image上的对应标号进行扩充:点膨胀法
    [OUT] seg_image 根据所在平面序号填充 seg_image
    @param point_enlarge: 一个点扩充单边扩充距离
*/
void DepthDetect::back_object_fill_2D(int point_enlarge)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            if (seg_image.at<uchar>(r, c) >= vp_start_no && seg_image.at<uchar>(r, c) < vp_no)
            {
                //因为下采样，补全扩充填充
                for (int i = r - point_enlarge; i <= r + point_enlarge; i++)
                {
                    if (i < 0)
                        continue;
                    if (i >= height)
                        break;
                    for (int j = c - point_enlarge; j <= c + point_enlarge; j++)
                    {
                        if (j < 0)
                            continue;
                        if (j >= width)
                            break;
                        seg_image.at<uchar>(i, j) = vp_no + (seg_image.at<uchar>(r, c) - vp_start_no); //用新标号标记,避免无限蔓延
                    }
                }
            }
        }
    }
}

/*
    对一个背景平面进行分割,粗修复法:矩形恢复
    [IN] 欧几里德聚类分割出的背景平面
    [OUT] seg_image 根据所在平面序号填充 seg_image
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
    @param inliers: 聚类中平面内点的索引，根据内点计算出矩形区域
*/
void DepthDetect::back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers)
{
    ObjectWindow object_window;
    float min_depth = FLT_MAX;
    //从聚类中提取出平面上的点,计算窗口
    for (const auto &idx : inliers->indices)
    {
        PointT border_point = (*cloud_cluster)[idx];
        object_window.add_point(border_point);
        //维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准
        if (border_point.z < min_depth)
            min_depth = border_point.z;
    }
    object_window.update();
    min_depths.push_back(min_depth);

    //遍历区域，将所有矩形区域内
    //行遍历,即使并行效率也没有明显提高，该部分应该耗时不大
    for (int r = object_window.topleft_x; r < object_window.topleft_x + object_window.height; r++)
    {
        //列遍历
        for (int c = object_window.topleft_y; c < object_window.topleft_y + object_window.width; c++)
        {
            if (seg_image.at<uchar>(r, c) == 0) //未被填充过
            {
                seg_image.at<uchar>(r, c) = vp_no;
            }
            else //已经被填充过,优先填充为深的
            {
                if (min_depth > min_depths[seg_image.at<uchar>(r, c) - vp_start_no]) //当前平面的深度较大
                    seg_image.at<uchar>(r, c) = vp_no;
            }
        }
    }
}

/*
    从前景点云中找出达到阈值的若干水平面，存储其点云以及平面系数,将平面按序号写入seg_image
    [in] cloud_foreground:从原始点云中按照深度阈值分割出的前景点云
    [out]  seg_image    将平面序号写入平面内点所属位置
    [out]  plane_clouds 将各平面内点集写入vector plane_clouds
    [out]  plane_coes   将各平面系数写入vector plane_coes(暂未使用)
    [out]  plane_border_clouds  将各平面边界内点集写入vector plane_border_clouds
    @param plane_seg_dis_threshold: (default=0.13f) 分割平面的距离阈值,前景精度较高应当合理控制,但不能过小，否则相当一部分会作为杂质残留，影响后续质量.输入同时存入类,提供给平面完整函数使用.
    @param layer_seg_dis_threshold: (default=0.3f)相同高度平面的同层分离的距离阈值(考虑相同高度多平面),目的是分割不靠近的同层平面，可以适当放宽---------------------------------此参数可能仍有待调整
*/
void DepthDetect::planar_seg(float _plane_seg_dis_threshold, float _layer_seg_dis_threshold)
{
    plane_seg_dis_threshold = _plane_seg_dis_threshold;
    layer_seg_dis_threshold = _layer_seg_dis_threshold;
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    // pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(plane_seg_dis_threshold);

    //用于暂存竖直面
    pcl::PointCloud<PointT>::Ptr cloud_vps(new pcl::PointCloud<PointT>());
    //平面遍历，按照阈值分割出若干平面，需要法线辨别，法线可以从平面系数计算，平面法向量：(A,B,C)。目的:找出支撑面识别并去除
    for (int i = 0;; i++)
    {
        seg.setInputCloud(cloud_foreground);
        seg.segment(*inliers, *coefficients);
        std::cout << "plane seg " << i << " inliers->indices.size()=" << inliers->indices.size() << ", fore_seg_threshold=" << fore_seg_threshold << std::endl;
        if (inliers->indices.size() < fore_seg_threshold)
        {
            break;
        }
        //分割出的平面可以判定为平面
        float A, D;
        A = coefficients->values[0];
        D = coefficients->values[3];

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_foreground);
        extract.setIndices(inliers);

        if (A >= 0.5) //初步判定平面为水平方向平面,水平面需要聚类，分割或者去除同一平面不相连区域。不能取绝对值
        {
            std::cout << "new ground coefficients: (" << coefficients->values[0] << ", " << coefficients->values[1] << ", " << coefficients->values[2] << ", " << coefficients->values[3] << ", " << std::endl;
            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_plane);

            if (abs(D) > max_D) //如果怀疑是地面，那就把整个平面暂存。如果有新的到来，把老的拿出来，新的存进去。
            {
                max_D = abs(D);
                if (ground_is_stored)
                {
                    pcl::PointCloud<PointT>::Ptr cloud_plane_g(new pcl::PointCloud<PointT>());
                    cloud_plane_g = ground_cloud;
                    ground_cloud = cloud_plane;
                    ground_coes = *coefficients; //存储地面模型系数，便于后续完整化
                    cloud_plane = cloud_plane_g; //有新的地面，把旧的拿出来做处理
                }
                else
                {
                    ground_cloud = cloud_plane;  //没有暂存，暂存该平面并跳过该平面的处理
                    ground_coes = *coefficients; //存储地面模型系数，便于后续完整化
                    ground_is_stored = true;
                    //---------Remove the planar inliers, extract the rest----------!!!
                    extract.setNegative(true);
                    extract.filter(*cloud_foreground);
                    continue;
                }
            }

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);

            //----------euclidean cluster extraction,相同高度可能存在多个平面，对同高度平面聚类分割不同平面------------------------
            //最好还是每次创建，如果多次重用同一个会导致未知问题，可能是没有回收，目前每次循环创建也可以，但不能保证更多的面不出问题。
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(layer_seg_dis_threshold);
            ec.setMinClusterSize(100); //如果限制该约束，较小的聚类会自动并入大的里面，追求的效果是将其返还
            tree->setInputCloud(cloud_plane);
            ec.setSearchMethod(tree);
            ec.setMaxClusterSize(cloud_plane->size() + 1);
            ec.setInputCloud(cloud_plane);
            ec.extract(cluster_indices);
            // std::cout << "plane " << i << " has " << cluster_indices.size() << " cluster" << std::endl;

            //--------------------遍历平面聚类，存储平面聚类点云、平面聚类参数、平面聚类边界点云，平面聚类分割结果---------------------------------
            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
            {
                // std::cout<<"it->indices.size() = "<<it->indices.size()<<", cloud_plane->size() * 0.2f="<<cloud_plane->size() * 0.2f<<std::endl;
                //如果聚类过小，比较可能是远处其他垂直面的一部分，将其返还给剩余点云
                if (it->indices.size() < cloud_plane->size() * 0.1f) //最小聚类数量：-------------------点云数量的二十分之一，或许过小了？？？
                {
                    for (const auto &idx : it->indices)
                        cloud_foreground->push_back((*cloud_plane)[idx]);
                }
                else //该聚类是一个独立平面,将其存储下来
                {
                    pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
                    for (const auto &idx : it->indices)
                        cloud_cluster->push_back((*cloud_plane)[idx]);
                    cloud_cluster->width = cloud_cluster->size();
                    cloud_cluster->height = 1;
                    cloud_cluster->is_dense = true;
                    if (!plane_check(cloud_cluster)) //如果不是平面,将其返回给剩余点云
                    {
                        cloud_foreground->concatenate(*cloud_foreground, *cloud_cluster);
                        continue;
                    }

                    //遍历,某高度平面内的一个独立平面的内点，不存在写冲突，即使已被写过，前景的优先级更高
                    for (auto &point : *cloud_cluster)
                    {
                        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                        int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                        seg_image.at<uchar>(r, c) = hp_no;
                    }
                    std::stringstream ss;
                    ss << "plane_cluster_" << hp_no << ".pcd";
                    pcl::io::savePCDFile(ss.str(), *cloud_cluster);

                    plane_border_clouds.push_back(*extract_border(cloud_cluster, 3));
                    plane_clouds.push_back(*cloud_cluster);
                    hp_no++; //存入了新的平面聚类,平面序号加一
                }
            }
        }
        else //不是水平面：忽略该平面，仍然要从剩余点云中去除，不然无法继续下一个平面。----需要做处理，不然垂面面直接没有了。可以考虑直接识别。暂时将其暂存并返回剩余点云
        {
            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_vp(new pcl::PointCloud<PointT>());
            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_vp);
            for (auto &point : *cloud_vp)
                cloud_vps->push_back(point);

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);
        }
    }
    //平面聚类识别结束，非地面平面聚类存入数据，平面仍在暂存中，将平面画到图上。
    if (ground_is_stored) //如果暂存非空，将暂存存入一般平面、边界
        ground_complete();
    //将竖面返回给剩余点云
    for (auto &point : *cloud_vps)
        cloud_foreground->push_back(point);
}
/*
    根据地面模型系数，从交界和背景中抽取数据，将地面完整化
    [IN] ground_coes 地面模型的系数。cloud_background cloud_junction
    [OUT] seg_image 将属于地面位置填充，ground_cloud 将地面点云完整化，cloud_foreground 地面非平面部分仍给前景。
*/
void DepthDetect::ground_complete()
{
    std::cout << std::endl
              << "ground_complete: ground coefficients: (" << ground_coes.values[0] << ", " << ground_coes.values[1] << ", " << ground_coes.values[2] << ", " << ground_coes.values[3] << ", " << std::endl;
    pcl::PointCloud<PointT>::Ptr cloud_background_plane(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_junction_plane(new pcl::PointCloud<PointT>);

    pcl::ModelOutlierRemoval<pcl::PointXYZ> back_ground_filter, junc_ground_filter;
    back_ground_filter.setModelCoefficients(ground_coes);
    back_ground_filter.setThreshold(plane_seg_dis_threshold * 3); //深度越大越杂乱,需要更大的阈值
    back_ground_filter.setModelType(pcl::SACMODEL_PLANE);
    back_ground_filter.setInputCloud(cloud_background);
    back_ground_filter.filter(*cloud_background_plane);
    back_ground_filter.setNegative(true);
    back_ground_filter.filter(*cloud_background);

    junc_ground_filter.setModelCoefficients(ground_coes);
    junc_ground_filter.setThreshold(plane_seg_dis_threshold * 2);
    junc_ground_filter.setModelType(pcl::SACMODEL_PLANE);
    junc_ground_filter.setInputCloud(cloud_junction);
    junc_ground_filter.filter(*cloud_junction_plane);
    junc_ground_filter.setNegative(true);
    junc_ground_filter.filter(*cloud_junction);

    // pcl::io::savePCDFile("ground.pcd", *ground_cloud);
    *ground_cloud += *cloud_background_plane;
    // pcl::io::savePCDFile("ground_back.pcd", *ground_cloud);
    *ground_cloud += *cloud_junction_plane;
    // pcl::io::savePCDFile("ground_back_junc.pcd", *ground_cloud);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(layer_seg_dis_threshold);
    ec.setMinClusterSize(100); //如果限制该约束，较小的聚类会自动丢弃?，追求的效果是将其返还
    tree->setInputCloud(ground_cloud);
    ec.setSearchMethod(tree);
    ec.setMaxClusterSize(ground_cloud->size() + 1);
    ec.setInputCloud(ground_cloud);
    ec.extract(cluster_indices);
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        for (const auto &idx : it->indices)
            cloud_cluster->push_back((*ground_cloud)[idx]);
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        if (!plane_check(cloud_cluster)) //如果不是平面,那么地面很可能是桌面，这个聚类是同高度其他物体的一部分。
        {
            cloud_foreground->concatenate(*cloud_foreground, *cloud_cluster);
            continue;
        }
        else
        {
            for (const auto &idx : it->indices)
            {
                PointT point = (*ground_cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                seg_image.at<uchar>(r, c) = ground_no;
            }
        }
    }
}
/*
    从点云聚类中拟合一条直线,如果大多数点都在直线内,这不是一个平面.
*/
bool DepthDetect::plane_check(pcl::PointCloud<PointT>::Ptr cloud_cluster, float is_line_threshold, float line_dis_threshold)
{
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(line_dis_threshold);

    seg.setInputCloud(cloud_cluster);
    seg.segment(*inliers, *coefficients);
    std::cout << "plane check: inliers->indices.size()=" << inliers->indices.size() << ", cloud_cluster->size()=" << cloud_cluster->size()
              << " *is_line_threshold=" << is_line_threshold * cloud_cluster->size() << std::endl;
    if (inliers->indices.size() > is_line_threshold * cloud_cluster->size())
        return false;
    else
        return true;
}

/*
    @param cloud_cluster: 分割出的平面聚类得到的独立平面的内点集合。无组织点云,从有组织一直提取过来的，实际上还是有原序的。
    @param n:   一个方向上一端提取点的数量
    @return  提取出来的边界点。
*/
pcl::PointCloud<PointT>::Ptr DepthDetect::extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n)
{
    pcl::PointCloud<PointT>::Ptr cloud_border(new pcl::PointCloud<PointT>);
    cv::Mat map = cv::Mat::zeros(height, width, CV_32F);

    int count = 0;
    //形成映射图像，使点云恢复组织性,便于边界提取
    for (auto &point : *cloud_cluster)
    {
        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
        int c = round(point.y * constant / point.z);
        map.at<float>(r, c) = point.z;
        count++;
    }

    std::deque<int> rc_idx_deque;
    //  横向提取两端边界点
    for (int r = 0; r < map.rows; r++)
    {
        for (int c = 0; c < map.cols; c++)
        {
            if (map.at<float>(r, c) != 0) //此处有点
            {
                rc_idx_deque.push_back(c);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.front();
                float z = map.at<float>(r, column);
                float x = r * z / constant;
                float y = column * z / constant;
                cloud_border->push_back(PointT(x, y, z));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.back();
                float z = map.at<float>(r, column);
                float x = r * z / constant;
                float y = column * z / constant;
                cloud_border->push_back(PointT(x, y, z));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }
    //  纵向提取两端边界点
    for (int c = 0; c < map.cols; c++)
    {
        for (int r = 0; r < map.rows; r++)
        {
            if (map.at<float>(r, c) != 0) //此处有点
            {
                rc_idx_deque.push_back(r);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.front();
                float z = map.at<float>(row, c);
                float x = row * z / constant;
                float y = c * z / constant;
                cloud_border->push_back(PointT(x, y, z));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.back();
                float z = map.at<float>(row, c);
                float x = row * z / constant;
                float y = c * z / constant;
                cloud_border->push_back(PointT(x, y, z));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }

    return cloud_border;
}

/*
    根据背景、平面、物体已经画在seg_image上的对应标号提取某标号边界
    @param seg_no: seg_image上的标号
    @param n:   一个方向上一端提取点的数量
    @return 二维的边界点，是图像上的点，用于opencv拟合
*/
std::vector<cv::Point> DepthDetect::extract_border_2D(int seg_no, int n)
{

    std::vector<cv::Point> border_points;

    std::deque<int> rc_idx_deque;
    //  横向提取两端边界点
    for (int r = 0; r < seg_image.rows; r++)
    {
        for (int c = 0; c < seg_image.cols; c++)
        {
            if (seg_image.at<uchar>(r, c) == seg_no) //此处有点
            {
                rc_idx_deque.push_back(c);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.front();
                border_points.push_back(cv::Point(column, r));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.back();
                border_points.push_back(cv::Point(column, r));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }
    //  纵向提取两端边界点
    for (int c = 0; c < seg_image.cols; c++)
    {
        for (int r = 0; r < seg_image.rows; r++)
        {
            if (seg_image.at<uchar>(r, c) == seg_no) //此处有点
            {
                rc_idx_deque.push_back(r);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.front();
                border_points.push_back(cv::Point(c, row));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.back();
                border_points.push_back(cv::Point(c, row));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }

    return border_points;
}

/*
    @param cloud_cluster: 分割出的平面聚类得到的独立平面的内点集合。无组织点云,从有组织一直提取过来的，实际上还是有原序的。
    @param n:   一个方向上一端提取点的数量
    @return 二维的边界点，是图像上的点，用于opencv拟合
*/
std::vector<cv::Point> DepthDetect::extract_border_2D_bak(pcl::PointCloud<PointT> cloud_cluster, int n)
{

    std::vector<cv::Point> border_points;
    cv::Mat map = cv::Mat::zeros(height, width, CV_8U);

    //形成映射图像，便于边界提取
    for (auto &point : cloud_cluster)
    {
        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
        int c = round(point.y * constant / point.z);
        map.at<uchar>(r, c) = 1;
    }

    std::deque<int> rc_idx_deque;
    //  横向提取两端边界点
    for (int r = 0; r < map.rows; r++)
    {
        for (int c = 0; c < map.cols; c++)
        {
            if (map.at<uchar>(r, c) == 1) //此处有点
            {
                rc_idx_deque.push_back(c);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.front();
                border_points.push_back(cv::Point(column, r));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int column = rc_idx_deque.back();
                border_points.push_back(cv::Point(column, r));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }
    //  纵向提取两端边界点
    for (int c = 0; c < map.cols; c++)
    {
        for (int r = 0; r < map.rows; r++)
        {
            if (map.at<uchar>(r, c) == 1) //此处有点
            {
                rc_idx_deque.push_back(r);
            }
        }
        for (int i = 0; i < n; i++)
        {
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.front();
                border_points.push_back(cv::Point(c, row));
                rc_idx_deque.pop_front();
            }
            if (rc_idx_deque.empty() == false)
            {
                int row = rc_idx_deque.back();
                border_points.push_back(cv::Point(c, row));
                rc_idx_deque.pop_back();
            }
        }
        rc_idx_deque.clear();
    }

    return border_points;
}

/*
    [in]: cloud_foreground,前景点云，此时为去除平面后的前景点云的剩余点云
    [out]: object_cloud | cloud_background | plane_clouds 将前景分为前景物体、背景物体、平面
*/
void DepthDetect::object_detect(float ec_dis_threshold)
{
    //-------------创建新点云，将剩余点云与桌面点云组合----------------------
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    int plane_num = plane_pure_border_clouds.size();

    std::vector<int> plane_idx_ends;
    //遍历平面边缘,将点云存入新点云并计算平面尾索引。平面0对应索引0
    for (int plane_no = 0; plane_no < plane_num; plane_no++)
    {
        //遍历平面边缘内点
        for (auto &point : plane_pure_border_clouds[plane_no])
        {
            cloud->push_back(point);
        }
        int plane_idx_end;
        if (plane_idx_ends.empty())
        {
            plane_idx_end = 0 + plane_pure_border_clouds[plane_no].size() - 1;
        }
        else
        {
            plane_idx_end = plane_idx_ends[plane_no - 1] + plane_pure_border_clouds[plane_no].size();
        }

        plane_idx_ends.push_back(plane_idx_end);
    }
    pcl::io::savePCDFile("border_cloud.pcd", *cloud);
    //将交界点云拷贝到新点云
    for (auto &point : *cloud_junction)
    {
        cloud->push_back(point);
    }
    pcl::io::savePCDFile("border_junc_cloud.pcd", *cloud);
    //确定交界点云的索引区域
    int junction_end_idx;
    if (plane_num != 0)
        junction_end_idx = plane_idx_ends[plane_idx_ends.size() - 1] + cloud_junction->size();
    else
        junction_end_idx = cloud_junction->size() - 1;

    //将前景剩余点云拷贝到新点云
    for (auto &point : *cloud_foreground)
        cloud->push_back(point);

    pcl::io::savePCDFile("border_junc_fore_cloud.pcd", *cloud);

    //-------------step1:对结合平面边缘的剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(ec_dis_threshold); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    //-------------step2:对不同聚类结果分别赋予不同的序列号(根据object_no)----------------
    // int object_num = 0;

    //遍历聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        int type = -1;
        bool include_plane_border = false;
        bool include_juction = false;
        if (plane_num != 0) //平面数量不等于零，检查物体聚类内是否有边界点，如果有，该类归属于边界
        {
            //遍历单个聚类内索引
            for (const auto &idx : it->indices)
            {
                //如果有索引属于平面边界,则该物体聚类包含平面平面,找它属于哪个边界,之后不再检查包含边界
                if (idx <= plane_idx_ends[plane_num - 1])
                {
                    if (!include_plane_border)
                    {
                        for (int plane_no = 0; plane_no < plane_num; plane_no++)
                            if (idx <= plane_idx_ends[plane_no])
                            {
                                type = plane_no;
                                include_plane_border = true;
                                break;
                            }
                    }
                }
                else if (idx <= junction_end_idx && !include_juction) ///如果有索引属于交界,该物体或平面被交界截开，之后不再检查是否包含交界
                {
                    if (type == -1) //同时包含交界和边界的情况type=palne_no
                        type = -2;
                    include_juction = true;
                }
                if (include_plane_border && include_juction) //同时包含边界点和交界点,判定此聚类是一个被截开的平面,检查结束
                    break;
            }
        }
        else //即使平面数量为零，仍要检查聚类内是否有点为交界点，如果有，设置标志为-2并在后续将聚类返还给背景点云
        {    //地面做了完整化处理，不会被截开，只可能是前景物体或背景物体
            for (const auto &idx : it->indices)
                if (idx <= junction_end_idx) //这个点属于交界点，将该物体归属于背景点云
                {
                    type = -2;
                    break;
                }
        }
        //该聚类中没有检测到边界点
        if (type == -1)
        {
            pcl::PointCloud<PointT>::Ptr object_cloud_cluster(new pcl::PointCloud<PointT>);
            //遍历聚类，修改聚类内点二维映射为物体编号。
            for (const auto &idx : it->indices)
                object_cloud_cluster->push_back((*cloud)[idx]);

            object_clouds.push_back(*object_cloud_cluster);
            //加入了一个新物体,序号加一
            object_no++;
        }
        else if (type == -2) //该物体聚类内有交界点返回给了背景
        {
            std::cout << "object_detect: return to cloud_background!" << std::endl;
            for (const auto &idx : it->indices)
                cloud_background->push_back((*cloud)[idx]);
        }
        else //该聚类中检测到了边界点,将其归属于平面type + hp_start_no
        {
            // pcl::PointCloud<PointT>::Ptr object_cloud_cluster(new pcl::PointCloud<PointT>);
            for (const auto &idx : it->indices)
            {
                // object_cloud_cluster->push_back((*cloud)[idx]);
                plane_clouds[type].push_back((*cloud)[idx]);
            }
            if (include_juction) //同时包含边界点与交界点，交界点部分单独存储，后续加入背景点云聚类，将后部平面加入平面
            {
                std::cout << "This is a break plane!" << std::endl; //暂不处理
            }
        }
    }
}

/*
    将桌子完整化，将桌子纯净边缘点加入剩余点云进行聚类，桌子边缘点所在聚类加入桌子。其他独立物体为聚类。
    [in] cloud_foreground: 去除支撑面的剩余点云
    [in] plane_pure_border_clouds
    [out] seg_image: 将支撑面完善成桌子等具体物体。
    @param ec_dis_threshold(default=0.25 0.12) 物体距离聚类阈值,应当适当小一些，独立部分较小会自动归属于附近类
    @param follow_point_enlarge 由于下采样点数不够稠密，但是桌腿一类直接凸包恢复影响形状，采用追随点增大策略，左右扩充数量
*/
void DepthDetect::object_detect_2D(float ec_dis_threshold, int follow_point_enlarge)
{
    //-------------创建新点云，将剩余点云与桌面点云组合----------------------
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    int plane_num = plane_pure_border_clouds.size();

    std::vector<int> plane_idx_ends;
    //遍历平面边缘,将点云存入新点云并计算平面尾索引。平面0对应索引0
    for (int plane_no = 0; plane_no < plane_num; plane_no++)
    {
        //遍历平面边缘内点
        for (auto &point : plane_pure_border_clouds[plane_no])
        {
            cloud->push_back(point);
        }
        int plane_idx_end;
        if (plane_idx_ends.empty())
        {
            plane_idx_end = 0 + plane_pure_border_clouds[plane_no].size() - 1;
        }
        else
        {
            plane_idx_end = plane_idx_ends[plane_no - 1] + plane_pure_border_clouds[plane_no].size();
        }

        plane_idx_ends.push_back(plane_idx_end);
    }
    //将交界点云拷贝到新点云
    for (auto &point : *cloud_junction)
    {
        cloud->push_back(point);
    }
    //确定交界点云的索引区域
    int junction_end_idx;
    if (plane_num != 0)
        junction_end_idx = plane_idx_ends[plane_idx_ends.size() - 1] + cloud_junction->size();
    else
        junction_end_idx = cloud_junction->size() - 1;

    //将前景剩余点云拷贝到新点云
    for (auto &point : *cloud_foreground)
    {
        cloud->push_back(point);
    }

    // std::stringstream ss0;
    // ss0 << "object_and_border.pcd";
    // pcl::io::savePCDFile(ss0.str(), *cloud);

    //-------------step1:对结合平面边缘的剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(ec_dis_threshold); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(cloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    //-------------step2:对不同聚类结果分别赋予不同的序列号(根据object_no)----------------
    // int object_num = 0;
    //遍历聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        int type = -1;
        bool include_plane_border = false;
        bool include_juction = false;
        if (plane_num != 0) //平面数量不等于零，检查物体聚类内是否有边界点，如果有，该类归属于边界
        {
            //遍历单个聚类内索引
            for (const auto &idx : it->indices)
            {
                //如果有索引属于平面边界,则该物体聚类包含平面平面,找它属于哪个边界,之后不再检查包含边界
                if (idx <= plane_idx_ends[plane_num - 1] && !include_plane_border)
                {
                    for (int plane_no = 0; plane_no < plane_num; plane_no++)
                        if (idx <= plane_idx_ends[plane_no])
                        {
                            type = plane_no;
                            include_plane_border = true;
                            break;
                        }
                }
                if (idx > plane_idx_ends[plane_num - 1] && idx <= junction_end_idx && !include_juction) ///如果有索引属于交界,该物体或平面被交界截开，之后不再检查是否包含交界
                {
                    if (type == -1) //同时包含交界和边界的情况type=palne_no
                        type = -2;
                    include_juction = true;
                }
                if (include_plane_border && include_juction) //同时包含边界点和交界点,判定此聚类是一个被截开的平面,检查结束
                    break;
            }
        }
        else //即使平面数量为零，仍要检查聚类内是否有点为交界点，如果有，设置标志为-2并在后续将聚类返还给背景点云
        {
            for (const auto &idx : it->indices)
            {
                if (idx <= junction_end_idx) //这个点属于交界点，将该物体归属于背景点云
                {
                    type = -2;
                    break;
                }
            }
        }

        //该聚类中没有检测到边界点,是物体，填入序号
        if (type == -1)
        {
            //遍历聚类，修改聚类内点二维映射为物体编号。
            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                seg_image.at<uchar>(r, c) = object_no;
            }
            // std::cout<<"object "<<object_no<<" num: "<<pn<<std::endl;
            //加入了一个新物体,序号加一
            object_no++;
        }
        else if (type == -2) //该物体聚类内只有交界点,返回给背景
        {
            // std::cout << "This object has junciton point, return it to background!" << std::endl;
            for (const auto &idx : it->indices)
            {
                cloud_background->push_back((*cloud)[idx]);
            }
        }
        else //该聚类中检测到了边界点
        {
            // std::cout << object_num << "This cluster is follow to plane" << type + hp_start_no << std::endl;
            //遍历聚类，修改聚类内点二维映射为平面编号。

            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                //因为下采样，补全扩充填充
                for (int i = r - follow_point_enlarge; i <= r + follow_point_enlarge; i++)
                {
                    if (i < 0)
                        continue;
                    if (i >= height)
                        break;
                    for (int j = c - follow_point_enlarge; j <= c + follow_point_enlarge; j++)
                    {
                        if (j < 0)
                            continue;
                        if (j >= width)
                            break;
                        seg_image.at<uchar>(i, j) = type + hp_start_no;
                    }
                }
            }
            if (include_juction) //同时包含边界点与交界点，交界点部分单独存储，后续加入背景点云聚类，将后部平面加入平面
            {
                std::cout << "This is a break plane!" << std::endl; //暂不处理
            }
            // std::cout << "object follow hp_no " << type + hp_start_no << std::endl;
        }
        // object_num++;
    }
}

/*
    净化识别出的平面的边界(不含地面)，通过拟合常见边界剔除因为遮挡造成的边界，目的在于使桌面桌腿连接，物体不因为与边界点相连而被归为平面
    如果可以拟合成椭圆，将椭圆内点识为纯净边界。如果可以拟合出两条及以上的直线对其边界做凸包并填充。[in] plane_border_clouds,只读不写
    [out] cloud_pure_border->plane_pure_border_clouds,为后续将边缘加入前景做准备
    @param fix 是否做平面形状修复
*/
void DepthDetect::caculate_clean_border(bool fix)
{

    int plane_num = plane_border_clouds.size();
    //循环遍历所有平面
    for (int plane_no = 0; plane_no < plane_num; plane_no++)
    {
        //-------------得到边界点-----------------
        pcl::PointCloud<PointT>::Ptr border_cloud(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr sor_border_cloud(new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr left_sor_border_cloud(new pcl::PointCloud<PointT>);
        for (auto &point : plane_border_clouds[plane_no])
            border_cloud->push_back(point); //遍历平面边缘内点

        std::stringstream ss1;
        ss1 << "plane_" << plane_no << "_border.pcd";
        pcl::io::savePCDFile(ss1.str(), *border_cloud);
        // Create the filtering object
        // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        // sor.setInputCloud(border_cloud);
        // sor.setMeanK(10);
        // sor.setStddevMulThresh(0.2);
        // sor.filter(*sor_border_cloud);
        // std::stringstream ss2;
        // ss2 << "plane_" << plane_no << "_sor_border.pcd";
        // pcl::io::savePCDFile(ss2.str(), *sor_border_cloud);

        //------------opencv二维角度拟合椭圆-------------------
        if (ellipse_fit(border_cloud, 0.5, 1.5, plane_no))
        {
            if (fix)
                shape_fix(plane_no);
        }
        else
        {
            int num = lines_fit(border_cloud, 0.1f, 0.1f, plane_no); // 2D sor_border_cloud, 0.1f, 0.1f, plane_no
            // std::cout << plane_no << "," << num << std::endl;
            if (num >= 2) //如果可以拟合出两条及以上的直线对其边界做凸包并填充。
            {
                if (fix)
                    shape_fix(plane_no);
            }
        }
    }
}

/*
    @brief 一个平面一个边界点云，一一对应，根据序号找到对应的边界点云，对边界点做凸包并填充包围区域
    [out] seg_image 用plane_no + hp_start_no填充seg_image
    @param plane_no: 平面的序号，据此找到其对应的平面聚类边界点云
*/
void DepthDetect::shape_fix(int plane_no)
{
    std::vector<cv::Point> contour;
    for (auto &point : plane_border_clouds[plane_no])
    {
        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
        int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
        contour.push_back(cv::Point(c, r));
    }
    // std::cout << contour.size() << std::endl;
    std::vector<cv::Point> convex;
    cv::convexHull(contour, convex);
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(convex);
    cv::fillPoly(seg_image, contours, cv::Scalar(plane_no + hp_start_no)); // fillPoly函数的第二个参数是二维数组！！
}
/*
    此前应先尝试拟合椭圆，不是椭圆则认为是多边形，从边界点中拟合出若干满足阈值的线段。
    [out] 将线段内点作为纯净边界点组合存储进plane_pure_border_clouds
    @param border_cloud: 边界点集合(非纯净的)
    @param line_threshold_percent: (default=0.2) 认定为直线的百分比阈值，默认占总边界点数的五分之一认定为是直线
    @param line_dis_threshold: (default=0.05) 拟合直线距离阈值，前景精度较高应当合理控制,但不能过小，否则相当一部分会无法识别
    @return line_num 该边界可以拟合出的直线的数量
*/
int DepthDetect::lines_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float line_threshold_percent, float line_dis_threshold, int plane_no)
{
    int border_point_num = border_cloud->size(); // border_cloud会过滤，数目会变化，因而应当在处理前先算。

    pcl::PointCloud<PointT>::Ptr cloud_pure_border(new pcl::PointCloud<PointT>()); //用于保存纯净的边界点

    int line_num = 0;
    while (true)
    {
        pcl::SACSegmentation<PointT> line_seg;
        pcl::PointIndices::Ptr line_inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr line_coefficients(new pcl::ModelCoefficients);
        pcl::PointCloud<PointT>::Ptr line_cloud_plane(new pcl::PointCloud<PointT>());

        line_seg.setOptimizeCoefficients(true);
        line_seg.setModelType(pcl::SACMODEL_LINE);
        line_seg.setMethodType(pcl::SAC_RANSAC);
        line_seg.setMaxIterations(100);
        line_seg.setDistanceThreshold(line_dis_threshold); //前景精度较高，计算圆的距离阈值应当合理控制,但不能过小，否则相当一部分会无法识别
        line_seg.setInputCloud(border_cloud);
        line_seg.segment(*line_inliers, *line_coefficients);

        if (line_inliers->indices.size() < border_point_num * line_threshold_percent)
        {
            plane_pure_border_clouds.push_back(*cloud_pure_border);
            break;
        }
        else
        {
            pcl::PointCloud<PointT>::Ptr cloud_line_border(new pcl::PointCloud<PointT>()); //用于暂存纯净的直线边界点
            pcl::ExtractIndices<PointT> line_extract;
            line_extract.setInputCloud(border_cloud);
            line_extract.setIndices(line_inliers);
            // Extract the line inliers from the input cloud
            line_extract.setNegative(false);
            line_extract.filter(*cloud_line_border);
            //将这条直线存入纯净边界。
            for (auto &point : *cloud_line_border)
                cloud_pure_border->push_back(point); //遍历平面边缘内点

            // std::stringstream ss1;
            // ss1 << "plane_" << plane_no << "_" << line_num << "_pure_border.pcd";
            // pcl::io::savePCDFile(ss1.str(), *cloud_line_border);

            //将直线从边界点云中去除
            line_extract.setNegative(true);
            line_extract.filter(*border_cloud);
            line_num++;
        }
    }
    return line_num;
}

/*
    @brief opencv二维角度拟合椭圆,[out]:如果成功将纯净内点组合存入plane_pure_border_clouds
    @param border_cloud: 边界点集合(非纯净的),只读不写
    @param fit_threshold_percent(default= 0.4f) 判断内点数是否达标的百分比阈值。
    @param dis_threshold (= 3.0f)   两点之间的距离阈值，用于判断是否椭圆内点，小于为内点
    @return 是否成功拟合椭圆。
*/
bool DepthDetect::ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float fit_threshold_percent, float dis_threshold, int plane_no)
{
    int inliers_num = 0;                                            //输入内点的数量
    int border_point_num = border_cloud->size();                    //边界点的总数
    float fit_threshold = fit_threshold_percent * border_point_num; //内点数是否达标阈值

    //将边界点转化为二维二值图像，便于二维拟合
    std::vector<cv::Point> border_points;
    // cv::Mat binary_image = cv::Mat::zeros(height, width, CV_8U);

    for (auto &point : *border_cloud)
    {
        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
        int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
        border_points.push_back(cv::Point(c, r));
        // binary_image.at<uchar>(r, c) = 100;
    }
    //用所给点拟合出了一个椭圆模型，参数返回
    cv::RotatedRect ellipse_rect = fitEllipse(border_points);
    //获取椭圆旋转角度。
    float angle = ellipse_rect.angle;
    float theta = angle * M_PI / 180.0; //角度转弧度
    //获取椭圆的长轴和短轴长度（外接矩形的长和宽）
    int w = ellipse_rect.size.width;
    int h = ellipse_rect.size.height;
    //椭圆的中心坐标
    cv::Point center = ellipse_rect.center;

    //平移旋转坐标系,两个容器坐标一一对应，因为要变换，将类型改变为float以精确计算。
    std::vector<cv::Point2f> transfer_border_points;
    for (auto &point : border_points)
    {
        cv::Point2f s_point, r_point;
        s_point.x = point.x - center.x;
        s_point.y = point.y - center.y;

        r_point.x = s_point.x * std::cos(theta) + s_point.y * std::sin(theta);
        r_point.y = -s_point.x * std::sin(theta) + s_point.y * std::cos(theta);

        transfer_border_points.push_back(cv::Point2f(r_point.x, r_point.y));
    }

    std::vector<int> pure_border_point_idxs;
    int border_point_no = 0;
    //遍历旋转平移后的坐标，在椭圆上找到他离最近的点，计算他与椭圆的距离--------事实上，该步骤可以与上个步骤合并
    for (auto &point : transfer_border_points)
    {
        cv::Point2f nearest_point = get_ellipse_nearest_point(w / 2, h / 2, point);
        float distance = std::sqrt(std::pow(nearest_point.x - point.x, 2) + std::pow(nearest_point.y - point.y, 2));
        if (distance < dis_threshold)
        {
            inliers_num++;
            pure_border_point_idxs.push_back(border_point_no);
        }
        border_point_no++;
    }
    std::cout << "ellipse fit: inliers_num=" << inliers_num << ", fit_threshold = " << fit_threshold << std::endl;
    if (inliers_num > fit_threshold)
    {
        //------------------extract pure border point to cloud_pure_border--------------------------
        pcl::PointCloud<PointT>::Ptr cloud_pure_border(new pcl::PointCloud<PointT>()); //用于保存纯净的边界点
        int border_point_idx = 0;
        int pure_border_point_no = 0;
        for (auto &point : *border_cloud) //所有的处理都是按照边界点的顺序执行的，因而匹配纯净点也可以按此顺序
        {
            if (border_point_idx == pure_border_point_idxs[pure_border_point_no])
            {
                cloud_pure_border->push_back(point);
                pure_border_point_no++;
            }
            border_point_idx++;
        }
        std::stringstream ss1;
        ss1 << "plane_" << plane_no << "_ellipse_pure_border.pcd";
        pcl::io::savePCDFile(ss1.str(), *cloud_pure_border);
        //---------------------store inlier points in plane_pure_border_clouds ---------------------------------
        plane_pure_border_clouds.push_back(*cloud_pure_border);
        return true;
    }
    else
    {
        return false;
    }
}

/*
    @brief 获取点p在椭圆上的最近点，用以计算距离。该算法基于未做旋转平移的坐标系，使用前使用旋转平移将椭圆移至合理位置。
    @param semi_major: x轴半轴长
    @param semi_minor: y轴半轴长
    @param p: 当前需计算距离点
    @return 椭圆上距离当前计算点最近的点
*/
cv::Point2f DepthDetect::get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p)
{
    //在尖端会受到影响,选择在第一象限进行计算，最终返回时加上符号
    float px = std::abs(p.x);
    float py = std::abs(p.y);

    //初始状态45度
    float t = M_PI / 4;

    float a = semi_major;
    float b = semi_minor;

    float x, y;

    for (int i = 0; i < 3; i++)
    {
        //(x,y)椭圆上的点
        x = a * std::cos(t);
        y = b * std::sin(t);

        //此处椭圆逼近圆的中心点
        float ex = (a * a - b * b) / a * std::pow(std::cos(t), 3);
        float ey = (b * b - a * a) / b * std::pow(std::sin(t), 3);
        //椭圆上点到圆心的向量
        float rx = x - ex;
        float ry = y - ey;
        //当前点到圆心的向量
        float qx = px - ex;
        float qy = py - ey;
        //向量长度
        float r = std::hypot(ry, rx);
        float q = std::hypot(qy, qx);

        // delta_t: t在椭圆上实现相同的弧长增量需要改变多少
        float delta_c = r * std::asin((rx * qy - ry * qx) / (r * q));
        float delta_t = delta_c / std::sqrt(a * a + b * b - x * x - y * y);

        t += delta_t;
        // t是负数则为0，t大于90度则为90度，t小于90度则为90度
        t = std::min((float)M_PI / 2, std::max(0.0f, t));
    }

    return cv::Point2f(copysign(x, p.x), copysign(y, p.y));
}

/*
    将由于边缘缺失导致的一个物体被分成两块的物体(碗等)进行合并
    [IN]    object_clouds 物体点云,随后被映射为二维空间的x\y坐标
    [OUT]   object_clouds 物体点云,部分物体加入其他物体,自身清空.获取结果为for循环形式,不用担心数组越界类似问题
    @param  merge_threshold: (default=0.8)上下边缘重合度达到阈值认定为同一物体，该阈值应该尽量大一些，否则易导致错误合并
*/
void DepthDetect::object_merge(float merge_threshold)
{
    //-------------遍历所有物体点云,检测x\y值域,注意此处坐标对照(点云中x为竖向)--------------------
    int object_num = object_no - object_start_no; // Object_no最终处于多加一状态
    std::vector<int> x_mins, y_mins;
    std::vector<int> x_maxs, y_maxs;
    for (int i = 0; i < object_num; i++)
    {
        std::vector<int> x_values, y_values;
        for (PointT point : object_clouds[i]) //必须转换,点云中的x\y与知觉分布是不一致的.
        {
            int y = round(point.x * constant / point.z); // grid_x = x * constant / depth
            int x = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
            x_values.push_back(x);
            y_values.push_back(y);
        }
        std::sort(x_values.begin(), x_values.end());
        std::sort(y_values.begin(), y_values.end());
        x_mins.push_back(x_values[0]);
        x_maxs.push_back(x_values[x_values.size() - 1]);
        y_mins.push_back(y_values[0]);
        y_maxs.push_back(y_values[y_values.size() - 1]);
    }
    //---------------检查水平值域重合度---------------------
    float **IOUs = new float *[object_num]; // p[]的元素为指针类型数据
    for (int i = 0; i < object_num; i++)
    {
        IOUs[i] = new float[object_num]; // p[i]为数组的函指针
    }

    for (int i = 0; i < object_num; i++)
    {
        // printf("%d x (%d,%d)",i,x_mins[i],x_maxs[i]);
        // printf("%d y (%d,%d)\n",i,y_mins[i],y_maxs[i]);
        for (int j = i + 1; j < object_num; j++)
        {
            int leni = x_maxs[i] - x_mins[i];
            int lenj = x_maxs[j] - x_mins[j];
            int len = leni > lenj ? leni : lenj;                       //最大长度
            int left = x_mins[i] > x_mins[j] ? x_mins[i] : x_mins[j];  //最大左端点
            int right = x_maxs[i] < x_maxs[j] ? x_maxs[i] : x_maxs[j]; //最小右端点
            IOUs[i][j] = (float)(right - left) / len;
            if (IOUs[i][j] < 0)
                IOUs[i][j] = 0;
            // printf("%d,%d: %.2f ",i,j,IOUs[i][j]);
        }
        // printf("\n");
    }

    std::vector<int> merge_tos;          //指示i号物体合并给了谁
    for (int i = 0; i < object_num; i++) //初始化
        merge_tos.push_back(i);

    for (int i = 0; i < object_num; i++)
    {
        for (int j = i + 1; j < object_num; j++)
        {
            if (IOUs[i][j] > 0.75f)
            {
                //水平方向值域重合度达标,判断边缘切合程度,注意避免同个物体多次合并找不到
                float left = y_mins[i] > y_mins[j] ? y_mins[i] : y_mins[j];  //最大左端点
                float right = y_maxs[i] < y_maxs[j] ? y_maxs[i] : y_maxs[j]; //最小右端点
                if (right - left >= 0)                                       // y值域有重合,合并
                {
                    // std::cout<<"merge"<<std::endl;
                    if (merge_tos[i] == i) //没被合并过,只需变自己
                    {
                        object_clouds[j] += object_clouds[merge_tos[i]];
                        object_clouds[merge_tos[i]].clear();
                        merge_tos[i] = j;
                    }
                    else //已经合并过了,还要再合并,把合并过的合并给新的,两个的编号都要更改
                    {
                        object_clouds[j] += object_clouds[merge_tos[i]];
                        object_clouds[merge_tos[i]].clear();
                        merge_tos[merge_tos[i]] = j;
                        merge_tos[i] = j;
                    }
                }
            }
        }
    }

    for (int i = 0; i < object_num; i++) //因为p是一个动态的数组，所以数组空间动态分配，程序不能自动 释放，所以自己要用delet释放。
    {
        delete[] IOUs[i];
        IOUs[i] = NULL;
    }
    delete[] IOUs; //释放指针和
    IOUs = NULL;
}

/*
    将由于边缘缺失导致的一个物体被分成两块的物体(碗)进行合并
    [IN]    seg_image   分割完物体的seg_image
    [OUT]   seg_image   物体合并后的seg_image
    @param  merge_threshold: (default=0.8)上下边缘重合度达到阈值认定为同一物体，该阈值应该尽量大一些，否则易导致错误合并
*/
void DepthDetect::object_merge_2D(float merge_threshold)
{
    //----------------从seg_image提取出每个物体x轴值域----------------------
    int object_num = object_no - object_start_no; // Object_no最终处于多加一状态
    std::vector<int> x_mins;
    std::vector<int> x_maxs;
    for (int i = 0; i < object_num; i++)
    {
        x_mins.push_back(width);
        x_maxs.push_back(0); //反向初始化
    }
    //遍历seg_image
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int seg_object_no = (int)seg_image.at<uchar>(r, c) - object_start_no; //这是第几个物体
            if (seg_object_no >= 0 && seg_object_no < object_num)
            {
                if (c < x_mins[seg_object_no])
                    x_mins[seg_object_no] = c;
                if (c > x_maxs[seg_object_no])
                    x_maxs[seg_object_no] = c;
            }
        }
    }

    float **IOUs = new float *[object_num]; // p[]的元素为指针类型数据
    for (int i = 0; i < object_num; i++)
    {
        IOUs[i] = new float[object_num]; // p[i]为数组的函指针
    }

    for (int i = 0; i < object_num; i++)
    {
        for (int j = i + 1; j < object_num; j++)
        {
            int leni = x_maxs[i] - x_mins[i];
            int lenj = x_maxs[j] - x_mins[j];
            int len = leni > lenj ? leni : lenj;                       //最大长度
            int left = x_mins[i] > x_mins[j] ? x_mins[i] : x_mins[j];  //最大左端点
            int right = x_maxs[i] < x_maxs[j] ? x_maxs[i] : x_maxs[j]; //最小右端点
            IOUs[i][j] = (float)(right - left) / len;
            if (IOUs[i][j] < 0)
                IOUs[i][j] = 0;
        }
    }
    for (int i = 0; i < object_num; i++)
    {
        for (int j = i + 1; j < object_num; j++)
        {
            if (IOUs[i][j] > 0.75) // 0.85?
            {
                // printf("%d,%d IOU %.2f merge\n", i, j, IOUs[i][j]);
                //水平方向值域重合度达标,判断边缘切合程度
                std::vector<cv::Point> borderi = extract_border_2D(object_start_no + i);
                std::vector<cv::Point> borderj = extract_border_2D(object_start_no + j);
                if (borderi.size() == 0 || borderj.size() == 0)
                    continue; //避免同个物体多次合并找不到
                std::vector<int> i_ys;
                std::vector<int> j_ys;
                for (cv::Point point : borderi)
                    i_ys.push_back(point.y);
                for (cv::Point point : borderj)
                    j_ys.push_back(point.y);
                std::sort(i_ys.begin(), i_ys.end());
                std::sort(j_ys.begin(), j_ys.end());
                int i_y_min = i_ys[0], i_y_max = i_ys[i_ys.size() - 1], j_y_min = j_ys[0], j_y_max = j_ys[j_ys.size() - 1];
                int left = i_y_min > j_y_min ? i_y_min : j_y_min;
                int right = i_y_max < j_y_max ? i_y_max : j_y_max;
                if (right - left >= -5)
                {
                    // std::cout<<"merge"<<std::endl;
                    for (int r = 0; r <= height; r++)
                        for (int c = x_mins[j]; c <= x_maxs[j]; c++)
                            if (seg_image.at<uchar>(r, c) == object_start_no + j)
                                seg_image.at<uchar>(r, c) = object_start_no + i;
                }
            }
        }
    }

    for (int i = 0; i < object_num; i++) //因为p是一个动态的数组，所以数组空间动态分配，程序不能自动 释放，所以自己要用delet释放。
    {
        delete[] IOUs[i];
        IOUs[i] = NULL;
    }
    delete[] IOUs; //释放指针和
    IOUs = NULL;
}

void DepthDetect::object_fill_2D()
{
    // std::vector<float> sorted_object_min_depths(object_points_nums);
    // std::sort(sorted_object_min_depths.begin(), sorted_object_min_depths.end());

    for (int j = object_start_no; j < object_no; j++)
    {
        // if (object_points_nums[j - object_start_no] < sorted_object_min_depths[sorted_object_min_depths.size() - 1])
        // {
        std::vector<cv::Point> contour = extract_border_2D(j);
        if (contour.size() != 0)
        {
            std::vector<cv::Point> convex;
            cv::convexHull(contour, convex);
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(convex);
            //填充区域之前，首先采用polylines()函数，可以使填充的区域边缘更光滑
            cv::polylines(seg_image, contours, true, cv::Scalar(j), 2, cv::LINE_8); //第2个参数可以采用contour或者contours，均可
            cv::fillPoly(seg_image, contours, cv::Scalar(j));                       // fillPoly函数的第二个参数是二维数组！！
        }
        // }
    }
}

void DepthDetect::plane_fill_2D()
{
    for (int i = hp_start_no; i < hp_no; i++)
    {
        std::vector<cv::Point> contour = extract_border_2D(i);
        if (contour.size() != 0)
        {
            std::vector<cv::Point> convex;
            cv::convexHull(contour, convex);
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(convex);
            cv::fillPoly(seg_image, contours, cv::Scalar(i)); // fillPoly函数的第二个参数是二维数组！！
        }
    }
    if (hp_no == hp_start_no) //没有识别出除地面外的平面
    {
        std::vector<cv::Point> contour = extract_border_2D(ground_no);
        if (contour.size() != 0)
        {
            std::vector<cv::Point> convex;
            cv::convexHull(contour, convex);
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(convex);
            cv::fillPoly(seg_image, contours, cv::Scalar(ground_no)); // fillPoly函数的第二个参数是二维数组！！
        }
    }
}
std::vector<cv::Rect> DepthDetect::get_object_window()
{
    int object_num = object_no - object_start_no;
    std::vector<cv::Rect> rects;
    std::vector<int> min_xs(object_num, width), max_xs(object_num, -1), min_ys(object_num, height), max_ys(object_num, -1);

    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int seg_object_no = (int)seg_image.at<uchar>(r, c) - object_start_no; //这是第几个物体
            if (seg_object_no >= 0 && seg_object_no < object_num)
            {
                if (c < min_xs[seg_object_no])
                {
                    min_xs[seg_object_no] = c;
                }
                if (c > max_xs[seg_object_no])
                {
                    max_xs[seg_object_no] = c;
                }
                if (r < min_ys[seg_object_no])
                {
                    min_ys[seg_object_no] = r;
                }
                if (r > max_ys[seg_object_no])
                {
                    max_ys[seg_object_no] = r;
                }
            }
        }
    }
    for (int i = 0; i < object_num; i++)
    {
        rects.push_back(cv::Rect(min_xs[i], min_ys[i], max_xs[i] - min_xs[i], max_ys[i] - min_ys[i]));
    }
    return rects;
}
std::vector<cv::Rect> DepthDetect::get_back_object_window()
{
    int back_object_num = vp_no - vp_start_no;
    std::vector<cv::Rect> rects;
    std::vector<int> back_min_xs(back_object_num, width), back_max_xs(back_object_num, -1), back_min_ys(back_object_num, height), back_max_ys(back_object_num, -1);
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {

            int seg_back_no = (int)seg_image.at<uchar>(r, c) - vp_start_no; //这是第几个背景物体
            if (seg_back_no >= 0 && seg_back_no < back_object_num)
            {
                if (c < back_min_xs[seg_back_no])
                {
                    back_min_xs[seg_back_no] = c;
                }
                if (c > back_max_xs[seg_back_no])
                {
                    back_max_xs[seg_back_no] = c;
                }
                if (r < back_min_ys[seg_back_no])
                {
                    back_min_ys[seg_back_no] = r;
                }
                if (r > back_max_ys[seg_back_no])
                {
                    back_max_ys[seg_back_no] = r;
                }
            }
        }
    }
    for (int i = 0; i < back_object_num; i++)
    {
        rects.push_back(cv::Rect(back_min_xs[i], back_min_ys[i], back_max_xs[i] - back_min_xs[i], back_max_ys[i] - back_min_ys[i]));
    }

    return rects;
}
std::vector<cv::Rect> DepthDetect::get_plane_window()
{
    int plane_num = hp_no - hp_start_no;
    std::vector<cv::Rect> rects;

    if (plane_num == 0)
    {
        int plane_min_xs = width, plane_max_xs = -1, plane_min_ys = height, plane_max_ys = -1;
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
                if (seg_image.at<uchar>(r, c) == 255)
                {
                    if (c < plane_min_xs)
                        plane_min_xs = c;
                    if (c > plane_max_xs)
                        plane_max_xs = c;
                    if (r < plane_min_ys)
                        plane_min_ys = r;
                    if (r > plane_max_ys)
                        plane_max_ys = r;
                }
        }
        rects.push_back(cv::Rect(plane_min_xs, plane_min_ys, plane_max_xs - plane_min_xs, plane_max_ys - plane_min_ys));
    }
    else
    {
        std::vector<int> plane_min_xs(plane_num, width), plane_max_xs(plane_num, -1), plane_min_ys(plane_num, height), plane_max_ys(plane_num, -1);
        for (int r = 0; r < height; r++)
        {
            for (int c = 0; c < width; c++)
            {
                int seg_back_no = (int)seg_image.at<uchar>(r, c) - hp_start_no; //这是第几个背景物体
                if (seg_back_no >= 0 && seg_back_no < plane_num)
                {
                    if (c < plane_min_xs[seg_back_no])
                        plane_min_xs[seg_back_no] = c;
                    if (c > plane_max_xs[seg_back_no])
                        plane_max_xs[seg_back_no] = c;
                    if (r < plane_min_ys[seg_back_no])
                        plane_min_ys[seg_back_no] = r;
                    if (r > plane_max_ys[seg_back_no])
                        plane_max_ys[seg_back_no] = r;
                }
            }
        }
        for (int i = 0; i < plane_num; i++)
        {
            rects.push_back(cv::Rect(plane_min_xs[i], plane_min_ys[i], plane_max_xs[i] - plane_min_xs[i], plane_max_ys[i] - plane_min_ys[i]));
        }
    }

    return rects;
}
