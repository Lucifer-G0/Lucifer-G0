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

	cv::String imagefolder = "../scene_01/*-depth.png";
	std::vector<std::string> image_paths;
	cv::glob(imagefolder, image_paths, false);

	//object-merge(object-merge opencv可能会产生莫名其妙创建错误)
	// #pragma omp parallel for
	for (auto image_path : image_paths)
	{
		int start = image_path.rfind("/"), end = image_path.rfind("-depth");
		start = start == std::string::npos ? 0 : start + 1;
		std::string image_no = image_path.substr(start, end - start);
		std::cout << "depth detect: " << image_no << std::endl;
		// std::string image_no="00035";
		// std::string image_path="../scene_01/00035-depth.png";

		DepthDetect dd(image_path, 2, 0.8f);
		dd.back_cluster_extract_2D();
		dd.planar_seg();
		dd.plane_fill_2D();
		dd.border_clean();//false
		dd.object_detect_2D(0.2,3);
		dd.object_merge();
		// dd.back_plane_fill_2D();
		dd.object_fill_2D();

		cv::Mat color_seg_image(dd.height, dd.width, CV_8UC3,cv::Scalar(0,0,0));
		dd.get_color_seg_image(color_seg_image);
		cv::imwrite("../output/"+image_no+"-seg.png", color_seg_image);
	// 	cv::imshow("object_detect_2D", color_seg_image);
	}

	std::cout << "[done, " << tt.toc () << " ms ]" << std::endl;


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