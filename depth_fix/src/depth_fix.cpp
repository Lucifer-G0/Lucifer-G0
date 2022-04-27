#include <opencv2/opencv.hpp>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

#include "DepthDetect.h"

#include <pcl/console/time.h>

int main()
{
	pcl::console::TicToc tt;
	tt.tic();

	DepthDetect dd;
	dd.back_cluster_extract();
	dd.planar_seg();
	dd.border_clean();
	dd.object_detect_2D();

	std::cout << "[done, " << tt.toc () << " ms ]" << std::endl;

	// cv::applyColorMap(dd.seg_image, color, cv::COLORMAP_HSV);
	cv::imshow("object_detect_2D", dd.get_color_seg());
	while (cv::waitKey(100) != 27)
	{
		if (cv::getWindowProperty("object_detect_2D", 0) == -1) //处理手动点击叉号关闭退出，报错退出，只能放在结尾
			break;
	}

	return 0;
}

void bak_code1()
{
	// 记录起始的时钟周期数
	double time = (double)cv::getTickCount();

	// 计算时间差
	time = ((double)cv::getTickCount() - time) / cv::getTickFrequency();

	// 输出运行时间
	std::cout << "运行时间：" << time << "秒\n";

	//------------------background result check-------------------------------
	// cv::Mat color;
	// cv::applyColorMap(dd.seg_image,color,cv::COLORMAP_HSV);
	// cv::imshow("object_detect_2D", color);
	// while (cv::waitKey(100) != 27)
	// {
	// 	if(cv::getWindowProperty("object_detect_2D",0) == -1)//处理手动点击叉号关闭退出，报错退出，只能放在结尾
	// 		break;
	// }
}