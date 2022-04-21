#include "Fix.h"
#include "ObjectWindow.h"

Fix::Fix(cv::Mat  _Depth)
{
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


//递归查找右邻居，找到非空或者边界停止，找到返回非空列号，找不到返回-1
int Fix::get_right_c(int r, int c)
{
	//当前点已经是最边界点，没有右邻居了
	if (++c >= width)
		return -1;
	//有右邻居，c已经是右邻居的行号，右邻居也是空的，接着往后找
	else if (Depth.at<uchar>(r, c) == 0)
		return get_right_c(r, c);
	//有右邻居并且右邻居不是空的。
	else
		return c;
}

int Fix::fix(int r, int c)
{
	return -1;
}

//[IN] 分割出的平面(达到深度阈值的)
void Fix::back_plane_fix(pcl::PointCloud<PointT>::Ptr cloud_cluster, pcl::PointIndices::Ptr inliers, pcl::ModelCoefficients::Ptr coefficients)
{

	ObjectWindow object_window;
	float A, B, C, D;
	A = coefficients->values[0];
	B = coefficients->values[1];
	C = coefficients->values[2];
	D = coefficients->values[3];

	std::cout << "Model: " << A << ", " << B << ", " << C << "," << D << std::endl;

	//从聚类中提取出平面上的点,计算窗口
	for (const auto &idx : inliers->indices)
	{
		PointT border_point = (*cloud_cluster)[idx];
		object_window.add_point(border_point);
	}

	object_window.update();
	object_window.output();
	object_window.draw(Depth);
	cv::imshow("window", Depth);
	cv::waitKey();
	//遍历区域，将所有矩形区域内深度修复,平面方程Ax+By+Cz+D=0->z
	//行遍历
	for (int r = object_window.topleft_x; r < object_window.topleft_x + object_window.height; r++)
	{
		//列遍历
		for (int c = object_window.topleft_y; c < object_window.topleft_y + object_window.width; c++)
		{
			if (Mask.at<uchar>(r, c) == 1 && Depth.at<float>(r, c) == 0) //是空洞点
			{
				//------------根据模型计算它的值
				float z = -D * constant * 1000. / (A * (r - 240.) + B * (c - 320.) + C * constant);
				// std::cout<<r<<","<<c<<","<<z<<std::endl;
				Depth.at<float>(r, c) = z; 
			}
			else if(Mask.at<uchar>(r, c) == 1)//已经被填充过的空洞点,优先填充为深的
			{
				float z = -D * constant * 1000. / (A * (r - 240.) + B * (c - 320.) + C * constant);
				if(z > Depth.at<float>(r, c))
				{
					Depth.at<float>(r, c) = z; 
				}
			}
		}
	}
	cv::imshow("fix_depth", Depth);
	cv::waitKey();
}