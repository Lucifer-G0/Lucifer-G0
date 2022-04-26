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
#include <math.h>

/*
    @param _cloud_foreground: 分割后的前景点云
    @param fore_seg_threshold_percent: 前景分割阈值百分比，判断分割出的平面中点的数量是否达标,用于控制分割结束
*/
ForeGround::ForeGround(pcl::PointCloud<PointT>::Ptr _cloud_foreground, float fore_seg_threshold_percent)
{
    cloud_foreground = _cloud_foreground;
    fore_seg_threshold = fore_seg_threshold_percent * cloud_foreground->size();
    hp_no = hp_start_no;
    object_no = object_start_no;
}

/*
    从前景点云中找出达到阈值的若干水平面，存储其点云以及平面系数,将平面按序号写入seg_image
    [in] cloud_foreground:从原始点云中按照深度阈值分割出的前景点云
    [out]  seg_image    将平面序号写入平面内点所属位置
    [out]  plane_clouds 将各平面内点集写入vector plane_clouds
    [out]  plane_coes   将各平面系数写入vector plane_coes
    [out]  plane_border_clouds  将各平面边界内点集写入vector plane_border_clouds
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
            std::cout << std::endl
                      << "This planar is too small!" << std::endl;
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

        //用于暂存竖直面
        pcl::PointCloud<PointT>::Ptr cloud_vps(new pcl::PointCloud<PointT>());
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

            // std::stringstream ss0;
            // ss0 << "cloud_plane_" << i << ".pcd";
            // pcl::io::savePCDFile(ss0.str(), *cloud_plane);

            //---------Remove the planar inliers, extract the rest----------
            extract.setNegative(true);
            extract.filter(*cloud_foreground);

            //----------euclidean cluster extraction------------------------
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

                    //遍历
                    for (auto &point : *cloud_cluster)
                    {
                        int r = point.x * constant / point.z; // grid_x = x * constant / depth
                        int c = point.y * constant / point.z;
                        seg_image.at<uchar>(r, c) = hp_no;
                    }
                    //一个物体处理结束,序号加一
                    hp_no++;

                    plane_border_clouds.push_back(*extract_border(cloud_cluster));
                    plane_clouds.push_back(*cloud_cluster);
                    plane_coes.push_back(*coefficients);

                    j++;
                }
            }
        }
        else //不是水平面：忽略该平面，仍然要从剩余点云中去除，不然无法继续下一个平面。----需要做处理，不然垂面面直接没有了。可以考虑直接识别。暂时将其暂存并返回剩余点云
        {
            std::cout << std::endl
                      << "A<0.5. This is not a horizontal plane!" << std::endl;

            // Extract the planar inliers from the input cloud
            pcl::PointCloud<PointT>::Ptr cloud_vp(new pcl::PointCloud<PointT>());
            //-------Get the points associated with the planar surface--------------
            extract.setNegative(false);
            extract.filter(*cloud_vp);
            for (auto &point : *cloud_vp)
                cloud_vps->push_back(point);
            std::cout << "Return it to foreground , num: " << cloud_vp->size() << std::endl
                      << std::endl;

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
void ForeGround::object_detect_2D_bak()
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

        // std::stringstream ss;
        // ss << "object_cluster_" << object_num++ << ".pcd";
        // pcl::io::savePCDFile(ss.str(), *cloud_cluster);

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

/*
    [in] cloud_foreground: 去除支撑面的剩余点云
    [in] plane_pure_border_clouds
    [out] seg_image: 将支撑面完善成桌子等具体物体。
    将桌子完整化，将桌子纯净边缘点加入剩余点云进行聚类，桌子边缘点所在聚类加入桌子。其他独立物体为聚类。
*/
void ForeGround::object_detect_2D()
{
    // 记录起始的时钟周期数
    double time = (double)cv::getTickCount();

    float ec_dis_threshold = 0.1; //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
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
            plane_idx_end = plane_idx_ends[plane_no - 1] + plane_pure_border_clouds[plane_no].size() - 1;
        }

        plane_idx_ends.push_back(plane_idx_end);
    }
    //将前景剩余点云拷贝到新点云
    for (auto &point : *cloud_foreground)
    {
        cloud->push_back(point);
    }

    //-------------step1:对结合平面边缘的剩余点云进行距离聚类-------------------------------
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(ec_dis_threshold); //物体距离聚类，阈值应当适当小一些，独立部分较小会自动归属于附近类
    ec.setMinClusterSize(30);
    ec.setMaxClusterSize(20000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
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
                    }
                }
            }
        }

        //该聚类中没有检测到边界点
        if (type == -1)
        {
            //遍历聚类，修改聚类内点二维映射为物体编号。
            for (const auto &idx : it->indices)
            {
                PointT point = (*cloud)[idx];
                int r = point.x * constant / point.z; // grid_x = x * constant / depth
                int c = point.y * constant / point.z;
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
                int r = point.x * constant / point.z; // grid_x = x * constant / depth
                int c = point.y * constant / point.z;
                seg_image.at<uchar>(r, c) = type + hp_start_no;
            }
        }
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "object_detect_2D 运行时间: " << time << "秒\n";
}

/*
    净化识别出的平面的边界，通过拟合常见边界剔除因为遮挡造成的边界，同时可以使物体不因为与边界点相连而被归为平面
    [in] plane_border_clouds
    [out] cloud_pure_border
*/
void ForeGround::border_clean()
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
        }
        else
        {
            lines_fit(border_cloud);
        }
    }

    // 计算时间差
    time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();
    // 输出运行时间
    std::cout << "border_clean 运行时间: " << time << "秒\n";
}
/*
    @brief 此前应先尝试拟合椭圆，不是椭圆则认为是多边形，从边界点中拟合出若干满足阈值的线段。
    @param border_cloud: 边界点集合(非纯净的)
    @param plane_no: 用于保存中间点云数据(拟合出的直线内点)
    @param [out] 将线段内点作为纯净边界点组合存储进plane_pure_border_clouds
*/
void ForeGround::lines_fit(pcl::PointCloud<PointT>::Ptr border_cloud, int plane_no)
{
    //阈值设置-------------------后续可能作为参数输入？
    float line_dis_threshold = 0.05; //前景精度较高，距离阈值应当合理控制,但不能过小，否则相当一部分会无法识别
    float line_threshold_percent = 0.2;

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
            std::cout << "This line is too small!" << std::endl;
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
}

void fg_store_code()
{
    // std::stringstream ss;
    // ss << "cloud_cluster_" << i << "_" << j << ".pcd";
    // pcl::io::savePCDFile(ss.str(), *cloud_cluster);
    // std::stringstream ss1;
    // ss1 << "cloud_border_" << i << "_"<< j << ".pcd";
    // pcl::io::savePCDFile(ss1.str(), *extract_border(cloud_cluster));

    // //------------先拟合圆形-------------------
    // pcl::SACSegmentation<PointT> cir_seg;
    // pcl::PointIndices::Ptr cir_inliers(new pcl::PointIndices);
    // pcl::ModelCoefficients::Ptr cir_coefficients(new pcl::ModelCoefficients);
    // pcl::PointCloud<PointT>::Ptr cir_cloud_plane(new pcl::PointCloud<PointT>());

    // cir_seg.setOptimizeCoefficients(true);
    // cir_seg.setModelType(pcl::SACMODEL_CIRCLE3D);
    // cir_seg.setMethodType(pcl::SAC_RANSAC);
    // cir_seg.setMaxIterations(100);
    // cir_seg.setDistanceThreshold(circle_dis_threshold); //前景精度较高，计算圆的距离阈值应当合理控制,但不能过小，否则相当一部分会无法识别
    // cir_seg.setInputCloud(border_cloud);
    // cir_seg.segment(*cir_inliers, *cir_coefficients);

    // //绘制拟合椭圆
    // cv::ellipse(binary_image, ellipse_rect, cv::Scalar(200), 1, 8);
    // cv::imshow("fitEllipse", binary_image);
    // while (true)
    // {
    //     if (cv::waitKey(30) == 27) //延时30ms,播放期间按下esc按键则退出，并返回键值
    //         break;
    // }
}

/*
    @brief opencv二维角度拟合椭圆,[out]:如果成功将纯净内点组合存入plane_pure_border_clouds
    @param border_cloud: 边界点集合(非纯净的)
    @return 是否成功拟合椭圆。
*/
bool ForeGround::ellipse_fit(pcl::PointCloud<PointT>::Ptr border_cloud)
{
    float dis_threshold = 3.0f;                                     //用于判断是否内点，两点之间的距离
    float fit_threshold_percent = 0.5f;                             //内点数是否达标阈值百分比。
    int inliers_num = 0;                                            //输入内点的数量
    int border_point_num = border_cloud->size();                    //边界点的总数
    float fit_threshold = fit_threshold_percent * border_point_num; //内点数是否达标阈值

    //将边界点转化为二维二值图像，便于二维拟合
    std::vector<cv::Point> border_points;
    // cv::Mat binary_image = cv::Mat::zeros(480, 640, CV_8U);

    for (auto &point : *border_cloud)
    {
        int r = point.x * constant / point.z; // grid_x = x * constant / depth
        int c = point.y * constant / point.z;
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
                  << std::endl;
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
cv::Point2f ForeGround::get_ellipse_nearest_point(float semi_major, float semi_minor, cv::Point2f p)
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
