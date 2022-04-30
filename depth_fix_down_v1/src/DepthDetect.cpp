#include <opencv2/opencv.hpp>
#include <math.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波器头文件

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

    // 创建滤波对象
    pcl::VoxelGrid<pcl::PointXYZ> filter;
    filter.setInputCloud(cloud);
    // 设置体素栅格的大小为 10x10x10cm
    filter.setLeafSize(0.1f, 0.1f, 0.1f);
    filter.filter(*cloud);

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
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(back_threshold, max_depth); //阈值到最大值
    pass.filter(*cloud_background);                  //背景点云
    pass.setFilterLimits(0.001, back_threshold - 0.001);
    pass.filter(*cloud_foreground); //前景点云,需注意前景必须去除零点，因为零点占相当大部分

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
                color_seg_image.at<cv::Vec3b>(r, c) = my_color.plane_colors[seg_no - hp_start_no];
            }
        }
    }
    return color_seg_image;
}

/*
    矩形修复，背景数据精度较低，比较杂乱，缺失较大，但总体上其位置分布相对集中，先聚类，然后采用较大的阈值在每个聚类内拟合出一个平面。
    [IN] cloud_background: 分割出的背景点云数据。
    [OUT]: Depth 每个聚类拟合一个矩形平面，将背景的空洞修复值填充到Depth | seg_image 根据所在平面序号填充 seg_image
    @param dimension: 维度(2 or 3) 2维在seg_image矩形填充平面序号，3维对深度数据Depth进行矩形区域恢复
    @param back_ec_dis_threshold: (default=0.5) 背景欧几里德聚类的距离阈值，背景数据精度低，相对杂乱，需要更大阈值，但不好控制
    @param plane_seg_dis_threshold: (default=0.2) 平面分割距离阈值
*/
void DepthDetect::back_cluster_extract(int dimension, float back_ec_dis_threshold, float plane_seg_dis_threshold)
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

        //---- Segment the largest planar component from the remaining cloud---------------
        seg.setInputCloud(cloud_cluster);
        seg.segment(*inliers, *coefficients);

        if (dimension == 2)
            back_plane_fix_2D_bak(cloud_cluster, inliers);
        else if (dimension == 3)
            back_plane_fix(cloud_cluster, inliers, coefficients);
        else
        {
            break;
        }
        //一个背景平面结束，背景平面序号加一
        vp_no++;
    }
}

/*
    对一个背景平面进行深度填充(点云层面、原深度图层面)
    [IN] 欧几里德聚类分割出的背景平面
    [OUT] Depth 根据所在平面系数将其矩形区域修复填充
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
    @param inliers: 聚类中平面内点的索引，根据内点计算出矩形区域
    @param coefficients: 该聚类内平面的系数，用于进行深度数据修复，在拟合平面上进行填充。
*/
void DepthDetect::back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
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
            }
            else if (Mask.at<uchar>(r, c) == 1) //已经被填充过的空洞点,优先填充为深的
            {
                float z = -D * constant * 1000. / (A * r + B * c + C * constant);
                if (z > Depth.at<float>(r, c))
                {
                    Depth.at<float>(r, c) = z;
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
    根据背景已经画在seg_image上的对应标号提取某标号边界
    [OUT] seg_image 根据所在平面序号填充 seg_image
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
*/
void DepthDetect::back_plane_fill_2D()
{
    for (int i = vp_start_no; i < vp_no; i++)
    {
        std::vector<cv::Point> contour = extract_border_2D(i);
        if (contour.size() != 0)
        {
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(contour);
            cv::fillPoly(seg_image, contours, cv::Scalar(i)); // fillPoly函数的第二个参数是二维数组！！
        }
    }
}

/*
    对一个背景平面进行分割,矩形恢复
    [IN] 欧几里德聚类分割出的背景平面
    [OUT] seg_image 根据所在平面序号填充 seg_image
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
    @param inliers: 聚类中平面内点的索引，根据内点计算出矩形区域
*/
void DepthDetect::back_plane_fix_2D_bak(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers)
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
    @param plane_seg_dis_threshold: (default=0.13f) 分割平面的距离阈值,前景精度较高应当合理控制,但不能过小，否则相当一部分会作为杂质残留，影响后续质量
    @param layer_seg_dis_threshold: (default=0.3f)相同高度平面的同层分离的距离阈值(考虑相同高度多平面),目的是分割不靠近的同层平面，可以适当放宽---------------------------------此参数可能仍有待调整
*/
void DepthDetect::planar_seg(float plane_seg_dis_threshold, float layer_seg_dis_threshold)
{
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
                    cloud_plane = cloud_plane_g;
                }
                else
                {
                    ground_cloud = cloud_plane; //没有暂存，暂存该平面并跳过该平面的处理
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

                    plane_border_clouds.push_back(*extract_border(cloud_cluster));
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
    {
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointT> ec;
        ec.setClusterTolerance(layer_seg_dis_threshold);
        ec.setMinClusterSize(100); //如果限制该约束，较小的聚类会自动并入大的里面，追求的效果是将其返还
        tree->setInputCloud(ground_cloud);
        ec.setSearchMethod(tree);
        ec.setMaxClusterSize(ground_cloud->size() + 1);
        ec.setInputCloud(ground_cloud);
        ec.extract(cluster_indices);
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
        {
            if (it->indices.size() < ground_cloud->size() * 0.1f)
                for (const auto &idx : it->indices)
                    cloud_foreground->push_back((*ground_cloud)[idx]);
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
    //将竖面返回给剩余点云
    for (auto &point : *cloud_vps)
        cloud_foreground->push_back(point);
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
    //形成映射图像，便于边界提取
    for (auto &point : *cloud_cluster)
    {
        int r = point.x * constant / point.z; // grid_x = x * constant / depth
        int c = point.y * constant / point.z;
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
        int r = point.x * constant / point.z; // grid_x = x * constant / depth
        int c = point.y * constant / point.z;
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
    [in]: cloud_foreground,前景点云，此时为去除平面垂面后的前景点云的剩余点云
    [out]: pcd 存储聚类结果到pcd
*/
void DepthDetect::object_detect()
{
    //-------------step1:对剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_foreground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.1); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(cloud_foreground->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_foreground);
    ec.extract(cluster_indices);

    //-------------step2:对不同聚类结果分别赋予不同的序列号(根据object_no)----------------
    //遍历聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        //-------------------------------------------此处可以直接改为赋值，不提取点云,但时间代价并不大
        for (const auto &idx : it->indices)
            cloud_cluster->push_back((*cloud_foreground)[idx]); //*
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        //一个物体处理结束,序号加一
        object_no++;
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
        if (plane_num == 0)
        {
            //遍历聚类，修改聚类内点二维映射为物体编号。
            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                seg_image.at<uchar>(r, c) = object_no;
            }

            //加入了一个新物体,序号加一
            object_no++;
            continue;
        }

        int type = -1;

        //遍历单个聚类内索引
        for (const auto &idx : it->indices)
        {
            //这个点属于平面边界,该聚类应该归属于边界。
            if (idx <= plane_idx_ends[plane_num - 1])
            {
                for (int plane_no = 0; plane_no < plane_num; plane_no++)
                {
                    if (idx <= plane_idx_ends[plane_no])
                    {
                        type = plane_no;
                        break;
                    }
                }
                break;
            }
        }

        //该聚类中没有检测到边界点
        if (type == -1)
        {
            int pn = 0;
            //遍历聚类，修改聚类内点二维映射为物体编号。
            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                seg_image.at<uchar>(r, c) = object_no;
                pn++;
            }
            // std::cout<<"object "<<object_no<<" num: "<<pn<<std::endl;

            //加入了一个新物体,序号加一
            object_no++;
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
            // std::cout << "object follow hp_no " << type + hp_start_no << std::endl;
        }
        // object_num++;
    }
}

/*
    净化识别出的平面的边界(不含地面)，通过拟合常见边界剔除因为遮挡造成的边界，目的在于使桌面桌腿连接，物体不因为与边界点相连而被归为平面
    如果可以拟合成椭圆，将椭圆内点识为纯净边界。如果可以拟合出两条及以上的直线对其边界做凸包并填充。
    [in] plane_border_clouds,只读不写   [out] cloud_pure_border
    @param fix 是否做平面形状修复
*/
void DepthDetect::border_clean(bool fix)
{

    int plane_num = plane_border_clouds.size();
    //循环遍历所有平面
    for (int plane_no = 0; plane_no < plane_num; plane_no++)
    {
        //-------------得到边界点-----------------
        pcl::PointCloud<PointT>::Ptr border_cloud(new pcl::PointCloud<PointT>);
        for (auto &point : plane_border_clouds[plane_no])
            border_cloud->push_back(point); //遍历平面边缘内点

        // std::stringstream ss0;
        // ss0 << "plane_" << plane_no << "_border.pcd";
        // pcl::io::savePCDFile(ss0.str(), *border_cloud);

        // /*方法三：统计滤波器滤波*/
        // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_after_StatisticalRemoval(new pcl::PointCloud<pcl::PointXYZ>); //
        // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Statistical;
        // Statistical.setInputCloud(border_cloud);
        // Statistical.setMeanK(5);           //取平均值的临近点数
        // Statistical.setStddevMulThresh(5); //临近点数数目少于多少时会被舍弃
        // Statistical.filter(*cloud_after_StatisticalRemoval);

        // std::stringstream ss1;
        // ss1 << "plane_" << plane_no << "_sr_border.pcd";
        // pcl::io::savePCDFile(ss1.str(), *cloud_after_StatisticalRemoval);

        //------------opencv二维角度拟合椭圆-------------------
        if (ellipse_fit(border_cloud))
        {
            if (fix)
                shape_fix(plane_no);
        }
        else
        {
            int num = lines_fit(border_cloud, 0.1f, 0.1f, plane_no);
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
bool DepthDetect::ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud, float fit_threshold_percent, float dis_threshold)
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
    将由于边缘缺失导致的一个物体被分成两块的物体(碗)进行合并
    [IN]    seg_image   分割完物体的seg_image
    [OUT]   seg_image   物体合并后的seg_image
    @param  merge_threshold: (default=0.8)上下边缘重合度达到阈值认定为同一物体，该阈值应该尽量大一些，否则易导致错误合并
*/
void DepthDetect::object_merge(float merge_threshold)
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
            if (IOUs[i][j] > 0.85)
            {
                // printf("%d,%d IOU %.2f merge\n", i, j, IOUs[i][j]);
                //水平方向值域重合度达标,判断边缘切合程度
                std::vector<cv::Point> borderi = extract_border_2D(object_start_no + i);
                std::vector<cv::Point> borderj = extract_border_2D(object_start_no + j);
                if(borderi.size()==0||borderj.size()==0) continue;//避免同个物体多次合并找不到
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
    for (int i = object_start_no; i < object_no; i++)
    {
        std::vector<cv::Point> contour = extract_border_2D(i);
        if (contour.size() != 0)
        {
            // std::vector<cv::Point> convex;
            // cv::convexHull(contour, convex);
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(contour);
            cv::fillPoly(seg_image, contours, cv::Scalar(i)); // fillPoly函数的第二个参数是二维数组！！
        }
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