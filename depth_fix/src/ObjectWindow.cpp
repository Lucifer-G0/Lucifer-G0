#include "ObjectWindow.h"

#include <iostream>

/*
    @brief when all points belong to this window are added, update this ObjectWindow
*/
void ObjectWindow::update()
{
    topleft_x=min_x;
    topleft_y=min_y;
    width=max_y-min_y+1;
    height=max_x-min_x+1;
}

/*
    @brief output object window (x,y,width,depth) to screen
*/
void ObjectWindow::output()
{
    
    std::cout << "object_window: (" << min_x << ", " << min_y << "), " << width << ", " << height << std::endl;
}

/*
    @brief draw object window to image (use image=object_window.draw(image);)
    @param  image  The image that the object window should draw.
    @return  The image of the object window has been drawn.
*/
cv::Mat ObjectWindow::draw(cv::Mat image)
{
    cv::rectangle(image, cv::Rect(topleft_y, topleft_x, width, height), cv::Scalar(50)); // cv::Scalar(B,G,R)
    return image;
}

/*
    @brief  Add point to this window. In fact,point is not add to it. Just update some info of this window. When add finish, use update()!!!

*/
void ObjectWindow::add_point(pcl::PointXYZ border_point)
{
    cv::Point point;                                            //特征点，用以画在图像中
    point.x = border_point.x * constant / border_point.z + 240; // grid_x = x * constant / depth
    point.y = border_point.y * constant / border_point.z + 320;

    if (point.x < min_x)
    {
        min_x = point.x;
    }
    if (point.x > max_x)
    {
        max_x = point.x;
    }
    if (point.y < min_y)
    {
        min_y = point.y;
    }
    if (point.y > max_y)
    {
        max_y = point.y;
    }
}