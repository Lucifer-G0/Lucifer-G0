#pragma once
#include <opencv2/opencv.hpp>

class MyCloud
{
public:
	cv::Mat X;
	cv::Mat Y;
	cv::Mat Z;
	int width;
	int height;
	MyCloud(cv::Mat _X, cv::Mat _Y, cv::Mat _Z, int w = 640, int h = 480) :X(_X), Y(_Y), Z(_Z), width(w), height(h) {}

};
