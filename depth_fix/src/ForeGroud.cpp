#include "ForeGround.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>

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
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.05); //前景精度较高，计算平面的距离阈值应当合理控制

    //---------prepare for euclidean cluster extraction---------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud_foreground);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.2);    //目的是分割不靠近的同层平面，可以适当放宽
    ec.setSearchMethod(tree);
    ec.setMinClusterSize(0); //如果限制该约束，较小的聚类会自动并入大的里面，追求的效果是将其返还

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
            std::cout << "A>=0.5. This is a horizontal plane！" << std::endl;
            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_plane);
            std::cout << "PointCloud representing the planar component: " << cloud_plane->size() << " data points." << std::endl;

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);

            //----------euclidean cluster extraction------------------------
            ec.setMaxClusterSize(cloud_plane->size());
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

                    plane_clouds.push_back(*cloud_cluster);
                    plane_coes.push_back(*coefficients);
                }
            }
        }
        else //不是水平面：忽略该平面，仍然要从剩余点云中去除，不然无法继续下一个平面。--------------------------需要做处理，不然垂面面直接没有了。可以考虑直接识别。
        {
            //----------------------------------------------------------------------------------------------------------
            //--------------------------Waiting for process---------------------------------------------------------
            //---------------------------------------------------------------------------------------
            std::cout << "A<0.5. This is not a horizontal plane！" << std::endl;
            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);
        }
    }
}