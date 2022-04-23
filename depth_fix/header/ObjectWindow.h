#ifndef _ObjectWindow_H_
#define _ObjectWindow_H_

#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <vtkType.h>
// opencv 画框使用的物体窗
class ObjectWindow
{
public:
    int topleft_x;
    int topleft_y;
    int width;
    int height;
    ObjectWindow():topleft_x(-1),topleft_y(-1),width(-1),height(-1){}
    ObjectWindow(int x, int y, int w, int h):topleft_x(x),topleft_y(y),width(w),height(h){}
    void output();
    void update();
    cv::Mat draw(cv::Mat image);
    void add_point(pcl::PointXYZ point);
private:
    // set min to MAX so that it will changed immediately when compare,原本是有正负号的，所以用有正负的float
    float min_x=VTK_FLOAT_MAX, min_y=VTK_FLOAT_MAX, max_x=VTK_FLOAT_MIN, max_y=VTK_FLOAT_MIN;
    float constant = 570.3;
};

#endif