
#include<iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <pcl/console/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

#include "MyCloud.h"

typedef pcl::PointXYZ PointT;

using namespace cv;

pcl::PointCloud<PointT>::Ptr depth2cloud(std::string filename, bool test)
{
	pcl::console::TicToc tt;
	Mat Depth = cv::imread(filename, -1);
	int imw = Depth.cols, imh = Depth.rows;
	int channels = Depth.channels();

	// Check whether it is a depth image.
	if (channels != 1)
	{
		if (channels == 3)
			std::cout << "this is a rgb image!" << std::endl;
		else
			std::cout << "channels = " << channels << std::endl;
	}
	if (test)
	{
		tt.tic();	// start timer
	}

	Depth.convertTo(Depth, CV_32F);

	// camera parameters
	float constant = 570.3, MM_PER_M = 1000;
	int change = constant * MM_PER_M;
	int center_x = imw/2, center_y = imh/2;

	// grid prepare
	Mat rangeW(1, imw, CV_32F), rangeH(1, imh, CV_32F);
	float* Wrowptr = rangeW.ptr<float>(0);
	float* Hrowptr = rangeH.ptr<float>(0);
	for (int i = 0; i < imw; i++) {
		Wrowptr[i] = i + 1;
	}
	for (int i = 0; i < imh; i++)
	{
		Hrowptr[i] = i + 1;
	}

	// x / gridx = depth / constant  �����ʵ��(���)�����������ı� ���� ʵ��������grid����ı� 
	Mat xgrid = Mat::ones(imh, 1, CV_32F) * rangeW - center_x;
	Mat ygrid = rangeH.t() * Mat::ones(1, imw, CV_32F) - center_y;

	Mat X = xgrid.mul(Depth) / change;
	Mat Y = ygrid.mul(Depth) / change;
	Mat Z = Depth / MM_PER_M;

	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
	cloud->width = imw;
	cloud->height = imh;
	
	for (int i = 0; i < imh; i++)
	{
		for (int j = 0; j < imw; j++)
		{
			float x = X.ptr<float>(i)[j];
			float y = Y.ptr<float>(i)[j];
			float z = Z.ptr<float>(i)[j];

			PointT point(x, y, z);
			cloud->push_back(point);
		}
	}
	
	if (!test)
	{
		// set cloud to organized
		cloud->width = imw;
		cloud->height = imh;
	}
	else
	{
		// // end timer and output used time
		// std::cout << "transform form depth to cloud cost:"<<tt.toc() << "ms" << std::endl << std::endl;
		// std::cout << "cloud size after input: " << cloud->size() << std::endl;
		// std::cout << "cloud orgnized after  input: " << cloud->isOrganized() << std::endl;

		// set cloud to organized
		cloud->width = imw;
		cloud->height = imh;

		// std::cout << "cloud size after reset w and h: " << cloud->size() << std::endl;
		// std::cout << "cloud orgnized after  reset w and h: " << cloud->isOrganized() << std::endl << std::endl;
		
		//remove suffix ".png"
		int start=filename.rfind("scene") , end= filename.rfind("-depth") ;
		std::string purename = filename.substr(start, end-start);

		//save cloud to pcd
		pcl::io::savePCDFile("../pcd/"+purename +"_cloud.pcd", *cloud);
		// std::cout << "save "<< purename <<"_cloud.pcd finish" << std::endl << std::endl;
	}
	

	return cloud;
}

pcl::PointCloud<PointT>::Ptr passthrough_filter(pcl::PointCloud<PointT>::Ptr cloud, float min, float max,bool test)
{
	pcl::PassThrough<PointT> pass;
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

	// Build a passthrough filter to remove spurious NaNs and scene background
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(min, max);
	pass.filter(*cloud_filtered);
	if (test)
	{
		std::cerr << "PointCloud after filtering has: " << cloud_filtered->size() << " data points." << std::endl;
		pcl::io::savePCDFile("../test_output/filtered/filtered.pcd", *cloud_filtered);
	}

	return cloud_filtered;
}

int png2video()
{
	cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);//���������־

	// cv.VideoWriter(	�ļ�����fourcc��fps��frameSize[��isColor]	) ->	<VideoWriter ����>
	VideoWriter video("test.avi", VideoWriter::fourcc('X', 'V', 'I', 'D'), 24.0, Size(640, 480));

	String img_path = "F:\\rgbd-scenes-v2_imgs\\rgbd-scenes-v2\\imgs\\scene_01";
	std::vector<String> img;

	cv::glob(img_path, img, false);

	size_t count = img.size();
	std::cout << count << std::endl;
	for (size_t i = 0; i < count; i++)
	{
		//cout << img[i] << endl;
		Mat image = imread(img[i]);
		if (!image.empty())
		{
			resize(image, image, Size(640, 480));
			video << image;
			std::cout << "���ڴ�����" << i << "֡" << std::endl;
		}
	}
	std::cout << "������ϣ�" << std::endl;

	return 0;
}

int video_show()
{
	cv::utils::logging::setLogLevel(utils::logging::LOG_LEVEL_SILENT);//���������־

//��
//utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);//ֻ���������־

	//����һ�����ڣ�������ʾ��ƵӰ��
	namedWindow("Example", WINDOW_AUTOSIZE);
	//VideoCapture ������������ƵӰ�����
	VideoCapture cap;
	//��һ��Ӱ������򿪳ɹ��Ļ�������ȡ��Ӱ�����Ϣ
	//C:/Users/GuSheng/Desktop/��׼����ͼƬ/768x576.avi
	bool bRet = cap.open("test.avi");
	//֡������
	int frames = cap.get(CAP_PROP_FRAME_COUNT);
	//֡�Ŀ���
	int iWidth = cap.get(CAP_PROP_FRAME_WIDTH);
	//֡�ĸ߶�
	int iHeight = cap.get(CAP_PROP_FRAME_HEIGHT);

	//�����Ӱ��ʧ�ܵĻ���ֱ�ӷ���
	if (!bRet)
	{
		return 1;
	}

	std::cout << frames << ", " << iWidth << ", " << iHeight << std::endl;

	//Mat ��������һ֡Ӱ�����Ϣ
	Mat img;
	//����ѭ����ʾӰ��
	while (true)
	{
		//img�д洢һ֡Ӱ����Ϣ
		cap >> img;
		//���û��Ӱ���ˣ�ֱ�ӷ���
		if (img.empty())
		{
			break;
		}
		//�ڴ�������ʾһ��Ӱ��
		imshow("Example", img);
		//�Ⱥ�33ms��������һ֡Ӱ��
		//waitkey���ذ�����code�����߷���-1����ָ����ʱ�䵽���ʱ��
		if (waitKey(33) >= 0)
		{
			break;
		}
	}
	return 0;
}

MyCloud depth2cloud_2(std::string filename)
{
	Mat Depth = cv::imread(filename, -1);
	int imw = Depth.cols, imh = Depth.rows;
	int channels = Depth.channels();

	// Check whether it is a depth image.
	if (channels != 1)
	{
		if (channels == 3)
			std::cout << "this is a rgb image!" << std::endl;
		else
			std::cout << "channels = " << channels << std::endl;
	}

	Depth.convertTo(Depth, CV_32F);

	// camera parameters
	float constant = 570.3, MM_PER_M = 1000;
	int change = constant * MM_PER_M;
	int center_x = 320, center_y = 240;

	// grid prepare
	Mat rangeW(1, imw, CV_32F), rangeH(1, imh, CV_32F);
	float* Wrowptr = rangeW.ptr<float>(0);
	float* Hrowptr = rangeH.ptr<float>(0);
	for (int i = 0; i < imw; i++) {
		Wrowptr[i] = i + 1;
	}
	for (int i = 0; i < imh; i++)
	{
		Hrowptr[i] = i + 1;
	}

	// x / gridx = depth / constant  �����ʵ��(���)�����������ı� ���� ʵ��������grid����ı� 
	Mat xgrid = Mat::ones(imh, 1, CV_32F) * rangeW - center_x;
	Mat ygrid = rangeH.t() * Mat::ones(1, imw, CV_32F) - center_y;

	Mat X = xgrid.mul(Depth) / change;
	Mat Y = ygrid.mul(Depth) / change;
	Mat Z = Depth / MM_PER_M;

	return MyCloud(X, Y, Z, imw, imh);
}