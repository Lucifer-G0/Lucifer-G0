#ifndef _ObjectWindow_H_
#define _ObjectWindow_H_

#include <opencv2/opencv.hpp>
// opencv 画框使用的物体窗
class ObjectWindow
{
public:
    int topleft_x;
    int topleft_y;
    int width;
    int height;
    ObjectWindow(int x, int y, int w, int h):topleft_x(x),topleft_y(y),width(w),height(h){}
    void output();
    cv::Mat draw(cv::Mat image);
};

#endif