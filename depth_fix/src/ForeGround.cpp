#include "ForeGround.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

/*
    @param _cloud_foreground: 分割后的前景点云
    @param fore_seg_threshold_percent: 前景分割阈值百分比，判断分割出的平面中点的数量是否达标,用于控制分割结束
*/
ForeGround::ForeGround(pcl::PointCloud<PointT>::Ptr _cloud_foreground, float fore_seg_threshold_percent)
{
    cloud_foreground = _cloud_foreground;
    fore_seg_threshold = fore_seg_threshold_percent * cloud_foreground->size();
}

/*
    从前景点云中找出达到阈值的若干水平面，存储其点云以及平面系数
*/
void ForeGround::planar_seg()
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
    seg.setDistanceThreshold(0.1); //前景精度较高，计算平面的距离阈值应当合理控制,但不能过小，否则相当一部分会作为杂质残留，影响后续质量

    //按照阈值分割出若干平面，需要法线辨别，法线可以从平面系数计算，平面法向量：(A,B,C)。目的:找出支撑面识别并去除
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
            std::cout << std::endl
                      << "A>=0.5. This is a horizontal plane!" << std::endl;
            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_plane);
            std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

            std::stringstream ss0;
            ss0 << "cloud_plane_" << i << ".pcd";
            pcl::io::savePCDFile(ss0.str(), *cloud_plane);

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);

            //----------euclidean cluster extraction------------------------
            //最好还是每次创建，如果多次重用同一个会导致未知问题，可能是没有回收，目前每次循环创建也可以，但不能保证更多的面不出问题。
            pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(0.3); //目的是分割不靠近的同层平面，可以适当放宽---------------------------------此参数可能仍有待调整
            ec.setMinClusterSize(3);     //如果限制该约束，较小的聚类会自动并入大的里面，追求的效果是将其返还
            tree->setInputCloud(cloud_plane);
            ec.setSearchMethod(tree);
            ec.setMaxClusterSize(cloud_plane->size() + 1);
            ec.setInputCloud(cloud_plane);
            ec.extract(cluster_indices);
            std::cout << "cluster_indices.size() = " << cluster_indices.size() << std::endl;

            //--------------------遍历聚类，每个聚类中找出一个平面，并用平面对矩形区域作修复---------------------------------
            int j = 0; //即使并行效率也没有明显提升,计算瓶颈不在这里。
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
                    std::stringstream ss;
                    ss << "cloud_cluster_" << i << "_" << j << ".pcd";
                    pcl::io::savePCDFile(ss.str(), *cloud_cluster);
                    // std::stringstream ss1;
                    // ss1 << "cloud_border_" << j << ".pcd";
                    // pcl::io::savePCDFile(ss1.str(), *extract_border(cloud_cluster));

                    plane_clouds.push_back(*cloud_cluster);
                    plane_coes.push_back(*coefficients);

                    j++;
                }
            }
        }
        else //不是水平面：忽略该平面，仍然要从剩余点云中去除，不然无法继续下一个平面。--------------------------需要做处理，不然垂面面直接没有了。可以考虑直接识别。
        {
            //----------------------------------------------------------------------------------------------------------
            //--------------------------Waiting for process---------------------------------------------------------
            //---------------------------------------------------------------------------------------
            std::cout << std::endl
                      << "A<0.5. This is not a horizontal plane!" << std::endl;
            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);
        }
    }

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
pcl::PointCloud<PointT>::Ptr ForeGround::extract_border(pcl::PointCloud<PointT>::Ptr cloud_cluster, int n)
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    pcl::PointCloud<PointT>::Ptr cloud_border(new pcl::PointCloud<PointT>);
    cv::Mat map = cv::Mat::zeros(480, 640, CV_32F);

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
std::vector<cv::Point> ForeGround::extract_border_2D(pcl::PointCloud<PointT> cloud_cluster, int n)
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    std::vector<cv::Point> border_points;
    cv::Mat map = cv::Mat::zeros(480, 640, CV_8U);

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
    [out]: write object serial number on "seg_image", which is a Mat storing segmentation result
*/
void ForeGround::object_detect_2D()
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    //-------------step1:对剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_foreground);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.05); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
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
    //遍历聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);
        //-------------------------------------------此处可以直接改为赋值，不提取点云
        for (const auto &idx : it->indices)
            cloud_cluster->push_back((*cloud_foreground)[idx]); //*
        cloud_cluster->width = cloud_cluster->size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        //遍历
        for (auto &point : *cloud_cluster)
        {
            int r = point.x * constant / point.z; // grid_x = x * constant / depth
            int c = point.y * constant / point.z;
            seg_image.at<uchar>(r, c) = object_no;
        }
        //一个物体处理结束,序号加一
        object_no++;
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "object_detect_2D 运行时间: " << time << "秒\n";
}