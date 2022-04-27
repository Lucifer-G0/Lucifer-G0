#include <opencv2/opencv.hpp>

// red
#define MY_Red cv::Vec3b(255, 0, 0)
#define MY_LightSalmon cv::Vec3b(255, 160, 122)
#define MY_Salmon cv::Vec3b(250, 128, 114)
#define MY_DarkSalmon cv::Vec3b(233, 150, 122)
#define MY_LightCoral cv::Vec3b(240, 128, 128)
#define MY_IndianRed cv::Vec3b(205, 92, 92)
#define MY_Crimson cv::Vec3b(220, 20, 60)
#define MY_FireBrick cv::Vec3b(178, 34, 34)
#define MY_DarkRed cv::Vec3b(139, 0, 0)
// pink
#define MY_Pink cv::Vec3b(255, 192, 203)
#define MY_LightPink cv::Vec3b(255, 182, 193)
#define MY_HotPink cv::Vec3b(255, 105, 180)
#define MY_DeepPink cv::Vec3b(255, 20, 147)
#define MY_PaleVioletRed cv::Vec3b(219, 112, 147)
#define MY_MediumVioletRed cv::Vec3b(199, 21, 133)
// orange
#define MY_Orange cv::Vec3b(255, 165, 0)
#define MY_DarkOrange cv::Vec3b(255, 140, 0)
#define MY_Coral cv::Vec3b(255, 127, 80)
#define MY_Tomato cv::Vec3b(255, 99, 71)
#define MY_OrangeRed cv::Vec3b(255, 69, 0)
// yellow
#define MY_Yellow cv::Vec3b(255, 255, 0)
#define MY_LightYellow cv::Vec3b(255, 255, 224)
#define MY_LemonChiffon cv::Vec3b(255, 250, 205)
#define MY_LightGoldenrodYellow cv::Vec3b(250, 250, 210)
#define MY_PapayaWhip cv::Vec3b(255, 239, 213)
#define MY_Moccasin cv::Vec3b(255, 228, 181)
#define MY_PeachPuff cv::Vec3b(255, 218, 185)
#define MY_PaleGoldenrod cv::Vec3b(238, 232, 170)
#define MY_Khaki cv::Vec3b(240, 230, 140)
#define MY_DarkKhaki cv::Vec3b(189, 183, 107)
#define MY_Gold cv::Vec3b(255, 215, 0)
// green
#define MY_Green cv::Vec3b(0, 128, 0)
#define MY_PaleGreen cv::Vec3b(152, 251, 152)
#define MY_LightGreen cv::Vec3b(144, 238, 144)
#define MY_YellowGreen cv::Vec3b(154, 205, 50)
#define MY_GreenYellow cv::Vec3b(173, 255, 47)
#define MY_Chartreuse cv::Vec3b(127, 255, 0)
#define MY_LawnGreen cv::Vec3b(124, 252, 0)
#define MY_Lime cv::Vec3b(0, 255, 0)
#define MY_LimeGreen cv::Vec3b(50, 205, 50)
#define MY_MediumSpringGreen cv::Vec3b(0, 250, 154)
#define MY_SpringGreen cv::Vec3b(0, 255, 127)
#define MY_MediumAquamarine cv::Vec3b(102, 205, 170)
#define MY_Aquamarine cv::Vec3b(127, 255, 212)
#define MY_LightSeaGreen cv::Vec3b(32, 178, 170)
#define MY_MediumSeaGreen cv::Vec3b(60, 179, 113)
#define MY_SeaGreen cv::Vec3b(46, 139, 87)
#define MY_DarkSeaGreen cv::Vec3b(143, 188, 143)
#define MY_ForestGreen cv::Vec3b(34, 139, 34)
#define MY_DarkGreen cv::Vec3b(0, 100, 0)
#define MY_OliveDrab cv::Vec3b(107, 142, 35)
#define MY_Olive cv::Vec3b(128, 128, 0)
#define MY_DarkOliveGreen cv::Vec3b(85, 107, 47)
#define MY_Teal cv::Vec3b(0, 128, 128)
// blue
#define MY_Blue cv::Vec3b(0, 0, 255)
#define MY_LightBlue cv::Vec3b(173, 216, 230)
#define MY_PowderBlue cv::Vec3b(176, 224, 230)
#define MY_PaleTurquoise cv::Vec3b(175, 238, 238)
#define MY_MediumTurquoise cv::Vec3b(72, 209, 204)
#define MY_Turquoise cv::Vec3b(64, 224, 208)
#define MY_DarkTurquoise cv::Vec3b(0, 206, 209)
#define MY_LightCyan cv::Vec3b(224, 255, 255)
#define MY_Cyan cv::Vec3b(0, 255, 255)
#define MY_Aqua cv::Vec3b(0, 255, 255)
#define MY_DarkCyan cv::Vec3b(0, 139, 139)
#define MY_CadetBlue cv::Vec3b(95, 158, 160)
#define MY_LightSteelBlue cv::Vec3b(176, 196, 222)
#define MY_SteelBlue cv::Vec3b(70, 130, 180)
#define MY_LightSkyBlue cv::Vec3b(135, 206, 250)
#define MY_SkyBlue cv::Vec3b(135, 206, 235)
#define MY_DeepSkyBlue cv::Vec3b(0, 191, 255)
#define MY_DodgerBlue cv::Vec3b(30, 144, 255)
#define MY_CornflowerBlue cv::Vec3b(100, 149, 237)
#define MY_RoyalBlue cv::Vec3b(65, 105, 225)
#define MY_MediumBlue cv::Vec3b(0, 0, 205)
#define MY_DarkBlue cv::Vec3b(0, 0, 139)
#define MY_Navy cv::Vec3b(0, 0, 128)
#define MY_MidnightBlue cv::Vec3b(25, 25, 112)
// brown
#define MY_Brown cv::Vec3b(165, 42, 42)
#define MY_Cornsilk cv::Vec3b(255, 248, 220)
#define MY_BlanchedAlmond cv::Vec3b(255, 235, 205)
#define MY_Bisque cv::Vec3b(255, 228, 196)
#define MY_NavajoWhite cv::Vec3b(255, 222, 173)
#define MY_Wheat cv::Vec3b(245, 222, 179)
#define MY_BurlyWood cv::Vec3b(222, 184, 135)
#define MY_Tan cv::Vec3b(210, 180, 140)
#define MY_RosyBrown cv::Vec3b(188, 143, 143)
#define MY_SandyBrown cv::Vec3b(244, 164, 96)
#define MY_Goldenrod cv::Vec3b(218, 165, 32)
#define MY_DarkGoldenrod cv::Vec3b(184, 134, 11)
#define MY_Peru cv::Vec3b(205, 133, 63)
#define MY_Chocolate cv::Vec3b(210, 105, 30)
#define MY_SaddleBrown cv::Vec3b(139, 69, 19)
#define MY_Sienna cv::Vec3b(160, 82, 45)
#define MY_Maroon cv::Vec3b(128, 0, 0)
// gray
#define MY_Gray cv::Vec3b(128, 128, 128)
#define MY_Gainsboro cv::Vec3b(220, 220, 220)
#define MY_LightGrey cv::Vec3b(211, 211, 211)
#define MY_Silver cv::Vec3b(192, 192, 192)
#define MY_DarkGray cv::Vec3b(169, 169, 169)
#define MY_DimGray cv::Vec3b(105, 105, 105)
#define MY_LightSlateGray cv::Vec3b(119, 136, 153)
#define MY_SlateGray cv::Vec3b(112, 128, 144)
#define MY_DarkSlateGray cv::Vec3b(47, 79, 79)
// purple
#define MY_Purple cv::Vec3b(128, 0, 128)
#define MY_Plum cv::Vec3b(221, 160, 221)
#define MY_Violet cv::Vec3b(238, 130, 238)
#define MY_Orchid cv::Vec3b(218, 112, 214)
#define MY_Fuchsia cv::Vec3b(255, 0, 255)
#define MY_Magenta cv::Vec3b(255, 0, 255)
#define MY_MediumOrchid cv::Vec3b(186, 85, 211)
#define MY_MediumPurple cv::Vec3b(147, 112, 219)
#define MY_Amethyst cv::Vec3b(153, 102, 204)
#define MY_BlueViolet cv::Vec3b(138, 43, 226)
#define MY_DarkViolet cv::Vec3b(148, 0, 211)
#define MY_DarkOrchid cv::Vec3b(153, 50, 204)
#define MY_DarkMagenta cv::Vec3b(139, 0, 139)

class MyColor
{
public:
    cv::Vec3b hole_color;
    cv::Vec3b ground_color;
    std::vector<cv::Vec3b> plane_colors;
    std::vector<cv::Vec3b> back_colors;
    std::vector<cv::Vec3b> object_colors;
    int pc_size;
    int bc_size;
    int oc_size;
    MyColor();
};

MyColor::MyColor()
{
    hole_color=cv::Vec3b(0, 0, 0);
    ground_color=MY_Purple;
    // purple to plane
    plane_colors.push_back(MY_Plum);
    plane_colors.push_back(MY_Violet);
    plane_colors.push_back(MY_Orchid);
    plane_colors.push_back(MY_MediumOrchid);
    plane_colors.push_back(MY_MediumPurple);
    plane_colors.push_back(MY_BlueViolet);
    plane_colors.push_back(MY_DarkViolet);
    plane_colors.push_back(MY_DarkMagenta);

    // gray to back
    back_colors.push_back(MY_Gray);
    back_colors.push_back(MY_LightGrey);
    back_colors.push_back(MY_Silver);
    back_colors.push_back(MY_DarkGray);
    back_colors.push_back(MY_DimGray);
    back_colors.push_back(MY_Gainsboro);
    back_colors.push_back(MY_LightSlateGray);
    back_colors.push_back(MY_SlateGray);
    back_colors.push_back(MY_DarkSlateGray);

    // others to object
    object_colors.push_back(MY_Red);
    object_colors.push_back(MY_LightSalmon);
    object_colors.push_back(MY_Salmon);
    object_colors.push_back(MY_DarkSalmon);
    object_colors.push_back(MY_LightCoral);
    object_colors.push_back(MY_IndianRed);
    object_colors.push_back(MY_Crimson);
    object_colors.push_back(MY_FireBrick);
    object_colors.push_back(MY_DarkRed);
    object_colors.push_back(MY_Pink);
    object_colors.push_back(MY_LightPink);
    object_colors.push_back(MY_HotPink);
    object_colors.push_back(MY_DeepPink);
    object_colors.push_back(MY_PaleVioletRed);
    object_colors.push_back(MY_MediumVioletRed);
    object_colors.push_back(MY_Orange);
    object_colors.push_back(MY_DarkOrange);
    object_colors.push_back(MY_Coral);
    object_colors.push_back(MY_Tomato);
    object_colors.push_back(MY_OrangeRed);
    object_colors.push_back(MY_Yellow);
    object_colors.push_back(MY_LightYellow);
    object_colors.push_back(MY_LemonChiffon);
    object_colors.push_back(MY_LightGoldenrodYellow);
    object_colors.push_back(MY_PapayaWhip);
    object_colors.push_back(MY_Moccasin);
    object_colors.push_back(MY_PeachPuff);
    object_colors.push_back(MY_PaleGoldenrod);
    object_colors.push_back(MY_Khaki);
    object_colors.push_back(MY_DarkKhaki);
    object_colors.push_back(MY_Gold);
    object_colors.push_back(MY_Green);
    object_colors.push_back(MY_PaleGreen);
    object_colors.push_back(MY_LightGreen);
    object_colors.push_back(MY_YellowGreen);
    object_colors.push_back(MY_GreenYellow);
    object_colors.push_back(MY_Chartreuse);
    object_colors.push_back(MY_LawnGreen);
    object_colors.push_back(MY_Lime);
    object_colors.push_back(MY_LimeGreen);
    object_colors.push_back(MY_MediumSpringGreen);
    object_colors.push_back(MY_SpringGreen);
    object_colors.push_back(MY_MediumAquamarine);
    object_colors.push_back(MY_Aquamarine);
    object_colors.push_back(MY_LightSeaGreen);
    object_colors.push_back(MY_MediumSeaGreen);
    object_colors.push_back(MY_SeaGreen);
    object_colors.push_back(MY_DarkSeaGreen);
    object_colors.push_back(MY_ForestGreen);
    object_colors.push_back(MY_DarkGreen);
    object_colors.push_back(MY_OliveDrab);
    object_colors.push_back(MY_Olive);
    object_colors.push_back(MY_DarkOliveGreen);
    object_colors.push_back(MY_Teal);
    object_colors.push_back(MY_Blue);
    object_colors.push_back(MY_LightBlue);
    object_colors.push_back(MY_PowderBlue);
    object_colors.push_back(MY_PaleTurquoise);
    object_colors.push_back(MY_MediumTurquoise);
    object_colors.push_back(MY_Turquoise);
    object_colors.push_back(MY_DarkTurquoise);
    object_colors.push_back(MY_LightCyan);
    object_colors.push_back(MY_Cyan);
    object_colors.push_back(MY_Aqua);
    object_colors.push_back(MY_DarkCyan);
    object_colors.push_back(MY_CadetBlue);
    object_colors.push_back(MY_LightSteelBlue);
    object_colors.push_back(MY_SteelBlue);
    object_colors.push_back(MY_LightSkyBlue);
    object_colors.push_back(MY_SkyBlue);
    object_colors.push_back(MY_DeepSkyBlue);
    object_colors.push_back(MY_DodgerBlue);
    object_colors.push_back(MY_CornflowerBlue);
    object_colors.push_back(MY_RoyalBlue);
    object_colors.push_back(MY_MediumBlue);
    object_colors.push_back(MY_DarkBlue);
    object_colors.push_back(MY_Navy);
    object_colors.push_back(MY_MidnightBlue);
    object_colors.push_back(MY_Brown);
    object_colors.push_back(MY_Cornsilk);
    object_colors.push_back(MY_BlanchedAlmond);
    object_colors.push_back(MY_Bisque);
    object_colors.push_back(MY_NavajoWhite);
    object_colors.push_back(MY_Wheat);
    object_colors.push_back(MY_BurlyWood);
    object_colors.push_back(MY_Tan);
    object_colors.push_back(MY_RosyBrown);
    object_colors.push_back(MY_SandyBrown);
    object_colors.push_back(MY_Goldenrod);
    object_colors.push_back(MY_DarkGoldenrod);
    object_colors.push_back(MY_Peru);
    object_colors.push_back(MY_Chocolate);
    object_colors.push_back(MY_SaddleBrown);
    object_colors.push_back(MY_Sienna);
    object_colors.push_back(MY_Maroon);

    pc_size=plane_colors.size();
    oc_size=object_colors.size();
    bc_size=back_colors.size();
}