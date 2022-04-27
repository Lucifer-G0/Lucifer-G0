#include "BackGround.h"
#include "ObjectWindow.h"
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

BackGround::BackGround(cv::Mat  _Depth)
{
	vp_no = vp_start_no;
	Depth=_Depth;
	width=Depth.cols;
	height=Depth.rows;
	cv::Mat _Mask(Depth.rows,Depth.cols,CV_8U);
	for (int r = 0; r < Depth.rows; r++)
	{
		for (int c = 0; c < Depth.cols; c++)
		{
			if (Depth.at<float>(r, c) == 0) //是空洞点
			{
				_Mask.at<uchar>(r, c) = 1; 
			}
			else
			{
				_Mask.at<uchar>(r, c) = 0; 
			}
		}
	}
	Mask=_Mask;

}

/*
	背景数据精度较低，比较杂乱，缺失较大，但总体上其位置分布相对集中，先聚类，然后采用较大的阈值在每个聚类内拟合出一个平面。
	[OUT]: Depth 每个聚类拟合一个矩形平面，将背景的空洞修复值填充到Depth | seg_image 根据所在平面序号填充 seg_image
	@param cloud_background: 分割出的背景点云数据。
	@param dimension: 维度(2 or 3) 2维在seg_image矩形填充平面序号，3维对深度数据Depth进行矩形区域恢复
*/
void BackGround::back_cluster_extract(pcl::PointCloud<PointT>::Ptr cloud_background,int dimension)
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

		if(dimension==2)
			back_plane_fix_2D(cloud_cluster, inliers);
		else if(dimension==3)
			back_plane_fix(cloud_cluster, inliers, coefficients);
		else
		{
			std::cout<<"This dimension is not right!"<<std::endl;
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
void BackGround::back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
{

	ObjectWindow object_window;
	float A, B, C, D;
	A = coefficients->values[0];
	B = coefficients->values[1];
	C = coefficients->values[2];
	D = coefficients->values[3];

	// std::cout << "Model: " << A << ", " << B << ", " << C << "," << D << std::endl;

	//从聚类中提取出平面上的点,计算窗口
	for (const auto &idx : inliers->indices)
	{
		PointT border_point = (*cloud_cluster)[idx];
		object_window.add_point(border_point);
	}

	object_window.update();
	// object_window.output();
	// object_window.draw(Depth);
	// cv::imshow("window", Depth);
	// cv::waitKey();

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
				float z = -D * constant * 1000. / (A *r + B * c+ C * constant);
				// std::cout<<r<<","<<c<<","<<z<<std::endl;
				Depth.at<float>(r, c) = z; 
			}
			else if(Mask.at<uchar>(r, c) == 1)//已经被填充过的空洞点,优先填充为深的
			{
				float z = -D * constant * 1000. / (A * r+ B * c + C * constant);
				if(z > Depth.at<float>(r, c))
				{
					Depth.at<float>(r, c) = z; 
				}
			}
		}
	}
	// cv::imshow("fix_depth", Depth);
	// cv::waitKey();
}

/*
	对一个背景平面进行分割
	[IN] 欧几里德聚类分割出的背景平面
	[OUT] seg_image 根据所在平面序号填充 seg_image
	@param cloud_cluster: 欧几里德聚类得到的一个聚类
	@param inliers: 聚类中平面内点的索引，根据内点计算出矩形区域
*/
void BackGround::back_plane_fix_2D(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers)
{
	ObjectWindow object_window;
	float min_depth=FLT_MAX;
	//从聚类中提取出平面上的点,计算窗口
	for (const auto &idx : inliers->indices)
	{
		PointT border_point = (*cloud_cluster)[idx];
		object_window.add_point(border_point);
		//维护一个平面序号对应的深度，从而实现距离深度优先，以聚类内最浅为标准
		if(border_point.z < min_depth)
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
				seg_image.at<uchar>(r,c)=vp_no;
			}
			else //已经被填充过,优先填充为深的
			{
				if(min_depth>min_depths[seg_image.at<uchar>(r, c)-vp_start_no])//当前平面的深度较大
					seg_image.at<uchar>(r,c)=vp_no;
			}
		}
	}

}