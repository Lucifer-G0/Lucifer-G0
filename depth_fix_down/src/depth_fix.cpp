#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>

#include "DepthDetect.h"

void detect_2D_Example();
void detect_3D_Example();

int main()
{
	
	detect_2D_Example();

	return 0;
}
void detect_3D_Example()
{
	pcl::console::TicToc tt;
	tt.tic();
	std::string image_no = "00000";
	std::string image_path = "../scene_01/00000-depth.png";
	std::cout << "depth detect: " << image_no << std::endl;

	DepthDetect dd(image_path, 3, 0.8f);
	dd.planar_seg();			//平面分割,平面点云存入plane_clouds,地面点云存入ground_cloud
	dd.caculate_clean_border(); //计算纯净边界,存入plane_pure_border_clouds,后续需加入前景聚类
	dd.object_detect(0.12f);			//前景物体检测，前景分为前景物体、背景和平面
	dd.object_merge();
	dd.back_cluster_extract(); //对背景做聚类，将背景分为背景物体
	std::cout << "[done, " << tt.toc() << " ms ]" << std::endl;
	dd.show_3D();
	// pcl::io::savePCDFile("color_pointcloud.pcd", *dd.get_color_pointcloud());
}
void detect_2D_Example()
{
	pcl::console::TicToc tt;
	tt.tic();

	cv::String imagefolder = "../scene_14/*-depth.png";
	std::vector<std::string> image_paths;
	cv::glob(imagefolder, image_paths, false);

	// object-merge(object-merge opencv可能会产生莫名其妙创建错误)
	//  #pragma omp parallel for
	for (auto image_path : image_paths)
	{
		int start = image_path.rfind("/"), end = image_path.rfind("-depth");
		start = start == std::string::npos ? 0 : start + 1;
		std::string image_no = image_path.substr(start, end - start);
		std::cout << "depth detect: " << image_no << std::endl;
		// std::string image_no="00318";
		// std::string image_path="../scene_11/00318-depth.png";

		DepthDetect dd(image_path, 2, 0.85f);

		dd.planar_seg();
		dd.plane_fill_2D();
		dd.caculate_clean_border();
		dd.object_detect_2D(0.2, 4);
		dd.object_merge_2D();
		dd.back_cluster_extract_2D();
		// dd.back_object_fill_2D(10);
		// dd.object_fill_2D();

		cv::Mat color_seg_image(dd.height, dd.width, CV_8UC3, cv::Scalar(0, 0, 0));
		dd.get_color_seg_image(color_seg_image);
		std::vector<cv::Rect> rects = dd.get_object_window();
		std::vector<cv::Rect> back_rects = dd.get_back_object_window();
		std::vector<cv::Rect> plane_rects = dd.get_plane_window();
		for (int i = 0; i < rects.size(); i++)
		{
			cv::rectangle(color_seg_image, rects[i], cv::Scalar(255, 0, 0), 3);
		}
		for (int i = 0; i < back_rects.size(); i++)
		{
			cv::rectangle(color_seg_image, back_rects[i], cv::Scalar(0, 255, 0), 1);
		}
		for (int i = 0; i < plane_rects.size(); i++)
		{
			cv::rectangle(color_seg_image, plane_rects[i], cv::Scalar(0, 0, 255), 1);
		}
		cv::imwrite("../output/" + image_no + "-seg.png", color_seg_image);
		// // 	cv::imshow("object_detect_2D", color_seg_image);
	}

	std::cout << "[done, " << tt.toc() << " ms ]" << std::endl;
}