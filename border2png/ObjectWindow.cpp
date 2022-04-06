#include <iostream>
#include "ObjectWindow.h"

/*
    @brief output object window (x,y,width,depth) to screen
*/
void ObjectWindow::output()
{
    std::cout<<"object_window: ("<<topleft_x<<", "<<topleft_y<<"), "<<width<<", "<<height<<std::endl;
}

/*
    @brief draw object window to image (use image=object_window.draw(image);)
    @param  image  The image that the object window should draw.
    @return  The image of the object window has been drawn.
*/
cv::Mat ObjectWindow::draw(cv::Mat image)
{ 
    cv::rectangle(image,cv::Rect(topleft_x,topleft_y,width,height),cv::Scalar(255,0,0)); // cv::Scalar(B,G,R)
    return image;
}