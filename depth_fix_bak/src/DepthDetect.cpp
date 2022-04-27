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

#include "DepthDetect.h"
#include "ObjectWindow.h"
#include "transform.h"
#include "MyColor.hpp"

DepthDetect::DepthDetect(int dimension)
{
    // 有些可以作为参数输入。
    float back_threshold_percent = 0.85f; //用于计算背景的深度阈值，百分比形式。0.85比较合适？
    float back_threshold = 0.0f;
    float max_depth = 50.0f;
    float fore_seg_threshold_percent = 0.1f; //前景分割是否平面阈值，前景点云大小的百分比
    std::string depth_path = "00179-depth.png";

    Depth = cv::imread(depth_path, -1);
    Depth.convertTo(Depth, CV_32F);
    width = Depth.cols;
    height = Depth.rows;

    Mask = cv::Mat(Depth.rows, Depth.cols, CV_8U);
    for (int r = 0; r < Depth.rows; r++)
        for (int c = 0; c < Depth.cols; c++)
            if (Depth.at<float>(r, c) == 0) //是空洞点
                Mask.at<uchar>(r, c) = 1;
            else
                Mask.at<uchar>(r, c) = 0;

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    cloud = depth2cloud(depth_path);
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
    pcl::io::savePCDFile("cloud_background.pcd", *cloud_background);
    pass.setFilterLimits(0.001, back_threshold - 0.001);
    pass.filter(*cloud_foreground); //前景点云,需注意前景必须去除零点，因为零点占相当大部分
    pcl::io::savePCDFile("cloud_foreground.pcd", *cloud_foreground);

    fore_seg_threshold = fore_seg_threshold_percent * cloud_foreground->size();

    hp_no = hp_start_no;
    object_no = object_start_no;
    vp_no = vp_start_no;
}

cv::Mat DepthDetect::get_color_seg()
{
    MyColor my_color;
    cv::Mat color_seg(480, 640, CV_8UC3);
    for (int r = 0; r < 480; r++)
    {
        for (int c = 0; c < 640; c++)
        {
            int seg_no = seg_image.at<uchar>(r, c);
            if (seg_no == 0)
            {
                color_seg.at<cv::Vec3b>(r, c) = my_color.hole_color;
            }
            else if (seg_no == 255)
            {
                color_seg.at<cv::Vec3b>(r, c) = my_color.ground_color;
            }
            else if (seg_no >= vp_start_no)
            {
                color_seg.at<cv::Vec3b>(r, c) = my_color.back_colors[((seg_no - vp_start_no)*3)%my_color.bc_size];
            }
            else if (seg_no >= object_start_no)
            {
                color_seg.at<cv::Vec3b>(r, c) = my_color.object_colors[((seg_no - object_start_no) * 7) % my_color.oc_size]; //对序号做变换实现相邻序号物体较大颜色跨度。
            }
            else if (seg_no >= hp_start_no)
            {
                color_seg.at<cv::Vec3b>(r, c) = my_color.plane_colors[seg_no - hp_start_no];
            }
        }
    }
    return color_seg;
}


/*
    背景数据精度较低，比较杂乱，缺失较大，但总体上其位置分布相对集中，先聚类，然后采用较大的阈值在每个聚类内拟合出一个平面。
    [IN] cloud_background: 分割出的背景点云数据。
    [OUT]: Depth 每个聚类拟合一个矩形平面，将背景的空洞修复值填充到Depth | seg_image 根据所在平面序号填充 seg_image
    @param dimension: 维度(2 or 3) 2维在seg_image矩形填充平面序号，3维对深度数据Depth进行矩形区域恢复
*/
void DepthDetect::back_cluster_extract(int dimension)
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
    seg.setDistanceThreshold(0.2);

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    //----------------------对背景聚类---------------------------------------------------------------也可以考虑先下采样再聚类？
    tree->setInputCloud(cloud_background);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(300);
    ec.setMaxClusterSize(20000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_background);
    ec.extract(cluster_indices);

    if (cluster_indices.size() == 0)
    {
        std::cout << "cluster_indices.size()==0" << std::endl;
    }

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
            std::cout << "This dimension is not right!" << std::endl;
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
                // std::cout<<r<<","<<c<<","<<z<<std::endl;
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
*/
void DepthDetect::back_cluster_extract_2D()
{
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    //----------------------对背景聚类---------------------------------------------------------------也可以考虑先下采样再聚类？
    tree->setInputCloud(cloud_background);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.5);
    ec.setMinClusterSize(300);
    ec.setMaxClusterSize(200000);
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
        back_plane_fix_2D(cloud_cluster);
        vp_no++;//一个聚类结束，背景平面序号加一
    }
}

/*
    对一个背景平面进行分割
    [IN] 欧几里德聚类分割出的背景平面
    [OUT] seg_image 根据所在平面序号填充 seg_image
    @param cloud_cluster: 欧几里德聚类得到的一个聚类
*/
void DepthDetect::back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud_cluster)
{
    std::vector<cv::Point> contour = extract_border_2D(*cloud_cluster);

    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(contour);
    //填充区域之前，首先采用polylines()函数，可以使填充的区域边缘更光滑
    cv::polylines(seg_image, contours, true, cv::Scalar(vp_no), 2, cv::LINE_AA); //第2个参数可以采用contour或者contours，均可
    cv::fillPoly(seg_image, contours, cv::Scalar(vp_no));                        // fillPoly函数的第二个参数是二维数组！！
}

/*
    对一个背景平面进行分割
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
    [out]  plane_coes   将各平面系数写入vector plane_coes
    [out]  plane_border_clouds  将各平面边界内点集写入vector plane_border_clouds
*/
void DepthDetect::planar_seg()
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.13); //前景精度较高，计算平面的距离阈值应当合理控制,但不能过小，否则相当一部分会作为杂质残留，影响后续质量

    //用于暂存竖直面
    pcl::PointCloud<PointT>::Ptr cloud_vps(new pcl::PointCloud<PointT>());
    //平面遍历，按照阈值分割出若干平面，需要法线辨别，法线可以从平面系数计算，平面法向量：(A,B,C)。目的:找出支撑面识别并去除
    for (int i = 0;; i++)
    {
        seg.setInputCloud(cloud_foreground);
        seg.segment(*inliers, *coefficients);
        std::cout << "inliers->indices.size():" << inliers->indices.size() << ", "
                  << "fore_seg_threshold:" << fore_seg_threshold << std::endl;

        if (inliers->indices.size() < fore_seg_threshold)
        {
            std::cout << "This planar is too small!" << std::endl;
            break;
        }
        //分割出的平面可以判定为平面
        float A, B, C, D;
        A = coefficients->values[0];
        B = coefficients->values[1];
        C = coefficients->values[2];
        D = coefficients->values[3];
        std::cout << "Model: " << A << ", " << B << ", " << C << "," << D << std::endl;

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud(cloud_foreground);
        extract.setIndices(inliers);

        if (A >= 0.5) //初步判定平面为水平方向平面,水平面需要聚类，分割或者去除同一平面不相连区域。
        {
            std::cout << "A>=0.5. This is a horizontal plane!" << std::endl;

            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_plane);
            std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

            if (abs(D) > max_D) //如果怀疑是地面，那就把整个平面暂存。如果有新的到来，把老的拿出来，新的存进去。
            {
                std::cout << "update ground: plane " << i << std::endl;
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
                    continue;
                }
            }

            std::stringstream ss0;
            ss0 << "cloud_plane_" << i << ".pcd";
            pcl::io::savePCDFile(ss0.str(), *cloud_plane);

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);

            //----------euclidean cluster extraction,相同高度可能存在多个平面，对同高度平面聚类分割不同平面------------------------
            //最好还是每次创建，如果多次重用同一个会导致未知问题，可能是没有回收，目前每次循环创建也可以，但不能保证更多的面不出问题。
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(0.3); //目的是分割不靠近的同层平面，可以适当放宽---------------------------------此参数可能仍有待调整
            ec.setMinClusterSize(30);    //如果限制该约束，较小的聚类会自动并入大的里面，追求的效果是将其返还
            tree->setInputCloud(cloud_plane);
            ec.setSearchMethod(tree);
            ec.setMaxClusterSize(cloud_plane->size() + 1);
            ec.setInputCloud(cloud_plane);
            ec.extract(cluster_indices);
            std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;

            //--------------------遍历平面聚类，存储平面聚类点云、平面聚类参数、平面聚类边界点云，平面聚类分割结果---------------------------------
            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
            {
                //如果聚类过小，比较可能是远处其他垂直面的一部分，将其返还给剩余点云
                if (it->indices.size() < cloud_plane->size() * 0.05f) //最小聚类数量：-------------------点云数量的二十分之一，或许过小了？？？
                {
                    std::cout << "This cluster is too small, return it to foreground " << it->indices.size() << std::endl;

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

                    std::cout << "This cluster is  not ground,  store plane " << hp_no << std::endl;
                    //遍历,某高度平面内的一个独立平面的内点，不存在写冲突，即使已被写过，前景的优先级更高
                    for (auto &point : *cloud_cluster)
                    {
                        int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                        int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                        seg_image.at<uchar>(r, c) = hp_no;
                    }
                    std::stringstream ss1;
                    ss1 << "cloud_plane_" << i << "cluster_" << hp_no << ".pcd";
                    pcl::io::savePCDFile(ss1.str(), *cloud_cluster);

                    plane_border_clouds.push_back(*extract_border(cloud_cluster));
                    plane_clouds.push_back(*cloud_cluster);
                    // plane_coes.push_back(*coefficients);
                    hp_no++; //存入了新的平面聚类,平面序号加一
                }
                std::cout << std::endl;
            }
        }
        else //不是水平面：忽略该平面，仍然要从剩余点云中去除，不然无法继续下一个平面。----需要做处理，不然垂面面直接没有了。可以考虑直接识别。暂时将其暂存并返回剩余点云
        {
            std::cout << "A<0.5. This is not a horizontal plane!" << std::endl;

            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_vp(new pcl::PointCloud<PointT>());
            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_vp);
            for (auto &point : *cloud_vp)
                cloud_vps->push_back(point);
            std::cout << "vp: temporarily store to cloud_vps , num: " << cloud_vps->size() << std::endl
                      << std::endl;

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);
        }
    }
    //平面聚类识别结束，非地面平面聚类存入数据，平面仍在暂存中，将平面画到图上。
    if (ground_is_stored) //如果暂存非空，将暂存存入一般平面、边界
    {
        std::cout << "平面聚类识别结束，非地面平面聚类存入数据，平面仍在暂存中，将平面画到图上。" << std::endl;
        for (auto &point : *ground_cloud)
        {
            int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
            int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
            seg_image.at<uchar>(r, c) = ground_no;
        }
    }
    //将竖面返回给剩余点云
    for (auto &point : *cloud_vps)
        cloud_foreground->push_back(point);
    std::cout << "Return cloud_vps to foreground , num: " << cloud_vps->size() << std::endl
              << std::endl;

    pcl::io::savePCDFile("fore_remove_support.pcd", *cloud_foreground);

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "planar_seg 运行时间: " << time << "秒\n\n";
}

/*
    @param cloud_cluster: 分割出的平面聚类得到的独立平面的内点集合。无组织点云,从有组织一直提取过来的，实际上还是有原序的。
    @param n:   一个方向上一端提取点的数量
    @return  提取出来的边界点。
*/
pcl::PointCloud<PointT>::Ptr DepthDetect::extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n)
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

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
    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();

    // 输出运行时间
    std::cout << "extract_border 运行时间: " << time << "秒\n";

    return cloud_border;
}

/*
    @param cloud_cluster: 分割出的平面聚类得到的独立平面的内点集合。无组织点云,从有组织一直提取过来的，实际上还是有原序的。
    @param n:   一个方向上一端提取点的数量
    @return 二维的边界点，是图像上的点，用于opencv拟合
*/
std::vector<cv::Point> DepthDetect::extract_border_2D(pcl::PointCloud<PointT> cloud_cluster, int n)
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

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

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "extract_border_2D 运行时间: " << time << "秒\n";

    return border_points;
}

/*
    [in]: cloud_foreground,前景点云，此时为去除平面垂面后的前景点云的剩余点云
    [out]: pcd 存储聚类结果到pcd
*/
void DepthDetect::object_detect()
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    //-------------step1:对剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_foreground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.1); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(20000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_foreground);
    ec.extract(cluster_indices);

    if (cluster_indices.size() == 0)
    {
        std::cout << "cluster_indices.size()==0" << std::endl;
    }

    //-------------step2:对不同聚类结果分别赋予不同的序列号(根据object_no)----------------
    int object_num = 0;
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

        std::stringstream ss;
        ss << "object_cluster_" << object_num++ << ".pcd";
        pcl::io::savePCDFile(ss.str(), *cloud_cluster);

        //一个物体处理结束,序号加一
        object_no++;
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "object_detect_2D 运行时间: " << time << "秒\n";
}

/*
    [in] cloud_foreground: 去除支撑面的剩余点云
    [in] plane_pure_border_clouds
    [out] seg_image: 将支撑面完善成桌子等具体物体。
    将桌子完整化，将桌子纯净边缘点加入剩余点云进行聚类，桌子边缘点所在聚类加入桌子。其他独立物体为聚类。
*/
void DepthDetect::object_detect_2D()
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    double ec_dis_threshold = 0.25; //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
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
    pcl::io::savePCDFile("000combine_pure_border_left.pcd", *cloud);

    //-------------step1:对结合平面边缘的剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(ec_dis_threshold); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(200000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    //-------------step2:对不同聚类结果分别赋予不同的序列号(根据object_no)----------------
    int object_num = 0;
    //遍历聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
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
        }
        else //该聚类中检测到了边界点
        {
            //遍历聚类，修改聚类内点二维映射为平面编号。
            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = round(point.x * constant / point.z); // grid_x = x * constant / depth
                int c = round(point.y * constant / point.z); //使用round实现四舍五入的float转int,默认的float转int只取整数位。
                seg_image.at<uchar>(r, c) = type + hp_start_no;
            }
            std::cout << "this cluster is follow to plane " << type << std::endl;
        }
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "object_detect_2D 运行时间: " << time << "秒\n";
}

/*
    净化识别出的平面的边界(不含地面)，通过拟合常见边界剔除因为遮挡造成的边界，目的在于使桌面桌腿连接，物体不因为与边界点相连而被归为平面
    如果可以拟合成椭圆，将椭圆内点识为纯净边界。如果可以拟合出两条及以上的直线对其边界做凸包并填充。
    [in] plane_border_clouds,只读不写   [out] cloud_pure_border
    @param fix 是否做平面形状修复
*/
void DepthDetect::border_clean(bool fix)
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    int plane_num = plane_border_clouds.size();
    //循环遍历所有平面
    for (int plane_no = 0; plane_no < plane_num; plane_no++)
    {
        //-------------得到边界点-----------------
        pcl::PointCloud<PointT>::Ptr border_cloud(new pcl::PointCloud<PointT>);
        for (auto &point : plane_border_clouds[plane_no])
            border_cloud->push_back(point); //遍历平面边缘内点
        std::cout << std::endl
                  << "total number of this border: " << border_cloud->size() << std::endl;

        //------border cloud save-------------
        std::stringstream ss;
        ss << "border_cloud_" << plane_no << ".pcd";
        pcl::io::savePCDFile(ss.str(), *border_cloud);

        //------------opencv二维角度拟合椭圆-------------------
        if (ellipse_fit(border_cloud))
        {
            if(fix)
                shape_fix(plane_no);
        }
        else
        {
            if (lines_fit(border_cloud, plane_no) >= 2) //如果可以拟合出两条及以上的直线对其边界做凸包并填充。
            {
                std::cout << "this border can fit 2 or more lines" << std::endl;
                if(fix)
                    shape_fix(plane_no);
            }
        }
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "border_clean 运行时间: " << time << "秒\n";
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
    std::vector<std::vector<cv::Point>> contours;
    contours.push_back(contour);
    //填充区域之前，首先采用polylines()函数，可以使填充的区域边缘更光滑
    cv::polylines(seg_image, contours, true, cv::Scalar(plane_no + hp_start_no), 2, cv::LINE_AA); //第2个参数可以采用contour或者contours，均可
    cv::fillPoly(seg_image, contours, cv::Scalar(plane_no + hp_start_no));                        // fillPoly函数的第二个参数是二维数组！！
}
/*
    此前应先尝试拟合椭圆，不是椭圆则认为是多边形，从边界点中拟合出若干满足阈值的线段。
    [out] 将线段内点作为纯净边界点组合存储进plane_pure_border_clouds
    @param border_cloud: 边界点集合(非纯净的)
    @param plane_no: 用于保存中间点云数据(拟合出的直线内点)
*/
int DepthDetect::lines_fit(pcl::PointCloud<PointT>::Ptr border_cloud, int plane_no)
{
    //阈值设置-------------------后续可能作为参数输入？
    float line_dis_threshold = 0.05;    //前景精度较高，距离阈值应当合理控制,但不能过小，否则相当一部分会无法识别
    float line_threshold_percent = 0.2; //占总边界点数的五分之一认定为是直线

    int border_point_num = border_cloud->size(); // border_cloud会过滤，数目会变化，因而应当在处理前先算。

    pcl::PointCloud<PointT>::Ptr cloud_pure_border(new pcl::PointCloud<PointT>()); //用于保存纯净的边界点

    int line_no = 0;
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

        std::cout << "fitting line inliers number: " << line_inliers->indices.size() << std::endl;

        if (line_inliers->indices.size() < border_point_num * line_threshold_percent)
        {
            std::cout << "This line is too small!" << std::endl; //如果直接就是这一步，那么没有经历提取自然不用返回。边界并有一一对应关系。一个平面最多一个纯净边界，没有也算个数。。
            std::cout << "push_back " << cloud_pure_border->size() << " points to plane_pure_border_clouds" << std::endl;
            plane_pure_border_clouds.push_back(*cloud_pure_border);
            break;
        }
        else
        {
            std::cout << "This is a line border!" << std::endl;
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

            // line cloud save
            std::stringstream ss_line;
            ss_line << "border_line_" << plane_no << "_" << line_no++ << ".pcd";
            pcl::io::savePCDFile(ss_line.str(), *cloud_line_border);

            //将直线从边界点云中去除
            line_extract.setNegative(true);
            line_extract.filter(*border_cloud);
        }
    }
    return line_no;
}

/*
    @brief opencv二维角度拟合椭圆,[out]:如果成功将纯净内点组合存入plane_pure_border_clouds
    @param border_cloud: 边界点集合(非纯净的),只读不写
    @return 是否成功拟合椭圆。
*/
bool DepthDetect::ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud)
{
    float dis_threshold = 3.0f;                                     //用于判断是否内点，两点之间的距离
    float fit_threshold_percent = 0.4f;                             //内点数是否达标阈值百分比。
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
    std::printf("angle,w,h,(center.x,center.y): %f, %d, %d, (%d,%d)\n", angle, w, h, center.x, center.y);

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
        // std::cout<<"point = "<<point<<", nearest_point = "<<nearest_point<<std::endl;
        float distance = std::sqrt(std::pow(nearest_point.x - point.x, 2) + std::pow(nearest_point.y - point.y, 2));
        // std::cout<<border_point_no<<" distance: "<<distance<<std::endl;
        if (distance < dis_threshold)
        {
            inliers_num++;
            pure_border_point_idxs.push_back(border_point_no);
        }
        border_point_no++;
    }

    if (inliers_num > fit_threshold)
    {
        std::cout << "This is  a ellipse, border_point_num = " << border_point_num << "inliers_num=" << inliers_num << std::endl;
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
        std::cout << "push_back " << cloud_pure_border->size() << " points to plane_pure_border_clouds" << std::endl
                  << std::endl;
        plane_pure_border_clouds.push_back(*cloud_pure_border);
        return true;
    }
    else
    {
        std::cout << "This is not a ellipse, border_point_num = " << border_point_num << "inliers_num=" << inliers_num << std::endl
                  << std::endl; //没有做提取操作，自然不用返回
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
