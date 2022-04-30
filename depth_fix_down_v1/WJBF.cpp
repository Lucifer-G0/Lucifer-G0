#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
cv::Mat get_Mask(cv::Mat Depth);
double space_weight(int current_x, int current_y, int neighboor_x, int neighboor_y, float delta_s = 23);
double value_weight(int current_value, int neighboor_value, float delta_c = 226);
double get_R(cv::Mat image, cv::Mat depth, cv::Mat Mask, int current_x, int current_y, int n=9, double delta_r = 47);
double get_K(cv::Mat image, cv::Mat depth, cv::Mat R, int current_x, int current_y, int n=9);
double get_fix_depth(cv::Mat image, cv::Mat depth, cv::Mat R, int current_x, int current_y, int n=9);


//获取高斯模板（空间模板）
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma);
//获取色彩模板（值域模板）
void getColorMask(std::vector<double> &colorMask , double colorSigma);
//双边滤波
void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma);

cv::Mat image = cv::imread("../scene_01/00000-color.png", 0);
cv::Mat Depth = cv::imread("../scene_01/00001-depth.png", -1); // use for WJBF



void BF()
{
	cv::Mat Depth = cv::imread("../scene_01/00001-depth.png", -1); 
    cv::Mat dst;
	cv::Size wsize(10,10);
    double spaceSigma=10;
    double colorSigma=35;

    bilateralfiter(Depth, dst,wsize,spaceSigma,colorSigma);

    imshow("Depth", Depth);
    imshow("Depth", dst);
    cv::waitKey(0);
}

 
//获取色彩模板（值域模板）
void getColorMask(std::vector<double> &colorMask , double colorSigma)
{
 
	for (int i = 0; i < 256; ++i){
		double colordiff = exp(-(i*i) / (2 * colorSigma * colorSigma));
		colorMask.push_back(colordiff);
	}
 
}
 
//获取高斯模板（空间模板）
void getGausssianMask(cv::Mat& Mask, cv::Size wsize, double spaceSigma)
{
	Mask.create(wsize, CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;
 
	for (int i = 0; i < h; ++i){
		y = pow(i - center_h, 2);
		double* Maskdate = Mask.ptr<double>(i);
		for (int j = 0; j < w; ++j){
			x = pow(j - center_w, 2);
			double g = exp(-(x + y) / (2 * spaceSigma * spaceSigma));
			Maskdate[j] = g;
			sum += g;
		}
	}
}

 
//双边滤波
void bilateralfiter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, double spaceSigma, double colorSigma)
{
	cv::Mat spaceMask;
	std::vector<double> colorMask;
	cv::Mat Mask0 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask1 = cv::Mat::zeros(wsize, CV_64F);
	cv::Mat Mask2 = cv::Mat::zeros(wsize, CV_64F);
 
	getGausssianMask(spaceMask, wsize, spaceSigma);//空间模板
	getColorMask(colorMask, colorSigma);//值域模板
	int hh = (wsize.height - 1) / 2;
	int ww = (wsize.width - 1) / 2;
	dst.create(src.size(), src.type());
	//边界填充
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, ww, ww, cv::BORDER_REPLICATE);//边界复制;
 
	for (int i = hh; i < src.rows + hh; ++i){
		for (int j = ww; j < src.cols + ww; ++j){
			double sum[3] = { 0 };
			int graydiff[3] = { 0 };
			double space_color_sum[3] = { 0.0 };
 
			for (int r = -hh; r <= hh; ++r){
				for (int c = -ww; c <= ww; ++c){
					if (src.channels() == 1){
						int centerPix = Newsrc.at<uchar>(i, j);
						int pix = Newsrc.at<uchar>(i + r, j + c);
						graydiff[0] = abs(pix - centerPix);
						double colorWeight = colorMask[graydiff[0]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight * spaceMask.at<double>(r + hh, c + ww);//滤波模板
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);
 
					}
					else if (src.channels() == 3){
						cv::Vec3b centerPix = Newsrc.at<cv::Vec3b>(i, j);
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c);
						graydiff[0] = abs(bgr[0] - centerPix[0]); graydiff[1] = abs(bgr[1] - centerPix[1]); graydiff[2] = abs(bgr[2] - centerPix[2]);
						double colorWeight0 = colorMask[graydiff[0]];
						double colorWeight1 = colorMask[graydiff[1]];
						double colorWeight2 = colorMask[graydiff[2]];
						Mask0.at<double>(r + hh, c + ww) = colorWeight0 * spaceMask.at<double>(r + hh, c + ww);//滤波模板
						Mask1.at<double>(r + hh, c + ww) = colorWeight1 * spaceMask.at<double>(r + hh, c + ww);
						Mask2.at<double>(r + hh, c + ww) = colorWeight2 * spaceMask.at<double>(r + hh, c + ww);
						space_color_sum[0] = space_color_sum[0] + Mask0.at<double>(r + hh, c + ww);
						space_color_sum[1] = space_color_sum[1] + Mask1.at<double>(r + hh, c + ww);
						space_color_sum[2] = space_color_sum[2] + Mask2.at<double>(r + hh, c + ww);
 
					}
				}
			}
 
			//滤波模板归一化
			if(src.channels() == 1)
			Mask0 = Mask0 / space_color_sum[0]; 
			else{
				Mask0 = Mask0 / space_color_sum[0];
				Mask1 = Mask1 / space_color_sum[1];
				Mask2 = Mask2 / space_color_sum[2];
			}
				
 
			for (int r = -hh; r <= hh; ++r){
				for (int c = -ww; c <= ww; ++c){
 
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * Mask0.at<double>(r + hh, c + ww); //滤波
					}
					else if(src.channels() == 3){
						cv::Vec3b bgr = Newsrc.at<cv::Vec3b>(i + r, j + c); //滤波
						sum[0] = sum[0] + bgr[0] * Mask0.at<double>(r + hh, c + ww);//B
						sum[1] = sum[1] + bgr[1] * Mask1.at<double>(r + hh, c + ww);//G
						sum[2] = sum[2] + bgr[2] * Mask2.at<double>(r + hh, c + ww);//R
					}
				}
			}
 
			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - ww) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b bgr = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i - hh, j - ww) = bgr;
			}
 
		}
	}
 
}



//空洞用Joint做了填充，不可以这么做，转点云完全混乱
void JBF()
{
    cv::Mat src = cv::imread("../scene_01/00000-depth.png", 1); // 原始带噪声的深度图
    cv::Mat joint = cv::imread("../scene_01/00000-color.png", 0);

    cv::Mat dst;
    int64 begin = cv::getTickCount();
    cv::ximgproc::jointBilateralFilter(joint, src, dst, -1, 3, 20);
    int64 end = cv::getTickCount();

    float time = (end - begin) / (cv::getTickFrequency() * 1000.);
    printf("time = %fms\n", time);

    imshow("src", src);
    imshow("joint", joint);
    imshow("jointBilateralFilter", dst);
    cv::waitKey(0);
}


// cost too much time
cv::Mat WJBF(cv::Mat joint,cv::Mat src)
{
    cv::Mat image = cv::imread("../scene_01/00000-color.png", 0);
    cv::Mat Depth = cv::imread("../scene_01/00001-depth.png", -1); // use for WJBF
    cv::Mat Mask=get_Mask(Depth);

    int64 begin = cv::getTickCount();
     for (int r = 0; r < Depth.rows; r++)
    {
        for (int c = 0; c < Depth.cols; c++)
        {
            if((int)Mask.at<uchar>(r,c)==0)
            {
                int fix_depth=get_fix_depth(image,Depth,Mask,r,c,30);
                // std::cout<<fix_depth<<std::endl;
                Depth.at<uchar>(r,c)=fix_depth;
            }
            
        }
    }
    int64 end = cv::getTickCount();
    float time = (end - begin) / (cv::getTickFrequency() * 1000.);
    printf("time = %fms\n", time);

    cv::imshow("fix",Depth);
    cv::waitKey(0);

    return Depth;
}

double space_weight(int current_x, int current_y, int neighboor_x, int neighboor_y, float delta_s)
{
    double distance = pow(pow(current_x - neighboor_x, 2) + pow(current_y - neighboor_y, 2), 0.5);
    return exp(-distance / 2 / delta_s);
}

double value_weight(int current_value, int neighboor_value, float delta_c)
{
    double distance = abs(current_value - neighboor_value);
    return exp(-distance / 2 / delta_c);
}

cv::Mat get_Mask(cv::Mat Depth)
{
    cv::Mat Mask(Depth.rows, Depth.cols, Depth.type());

    for (int r = 0; r < Depth.rows; r++)
    {
        for (int c = 0; c < Depth.cols; c++)
        {
            if (Depth.at<uchar>(r, c) == 0)
                Mask.at<uchar>(r, c) = (uchar)0;
            else
                Mask.at<uchar>(r, c) = (uchar)1;
        }
    }

    return Mask;
}

// n*n search
double get_R(cv::Mat image, cv::Mat depth, cv::Mat Mask, int neighboor_x, int neighboor_y, int n, double delta_r)
{
    double r = 0;

    int topleft_x = neighboor_x - (n - 1) / 2, topleft_y = neighboor_y - (n - 1) / 2;
    for (int x = topleft_x; x < topleft_x + n; x++)
    {
        if (x < 0)
            continue;
        else if (x >= depth.rows)
            break;
        for (int y = topleft_y; y < topleft_y + n; y++)
        {
            if (y < 0)
                continue;
            else if (y >= depth.cols)
                break;

            int mask = (int)Mask.at<uchar>(x, y);
            double s_w = space_weight(neighboor_x, neighboor_y, x, y);
            double v_w = value_weight((int)image.at<uchar>(neighboor_x, neighboor_y), (int)image.at<uchar>(x, y));
            int distance = abs((int)depth.at<uchar>(neighboor_x, neighboor_y) - (int)depth.at<uchar>(x, y));

            r += mask * s_w * v_w * exp(-distance / 2 / delta_r);
        }
    }

    return r;
}

double get_K(cv::Mat image, cv::Mat depth, cv::Mat Mask, int current_x, int current_y, int n)
{
    double k = 0;

    int topleft_x = current_x - (n - 1) / 2, topleft_y = current_y - (n - 1) / 2;
    for (int x = topleft_x; x < topleft_x + n; x++)
    {
        if (x < 0)
            continue;
        else if (x >= depth.rows)
            break;
        for (int y = topleft_y; y < topleft_y + n; y++)
        {
            if (y < 0)
                continue;
            else if (y >= depth.cols)
                break;

            double s_w = space_weight(current_x, current_y, x, y);
            double v_w = value_weight((int)image.at<uchar>(current_x, current_y), (int)image.at<uchar>(x, y));
            double r = get_R(image,depth,Mask,x,y);
            k += s_w * v_w * r;
        }
    }

    return k;
}

double get_fix_depth(cv::Mat image, cv::Mat depth, cv::Mat Mask, int current_x, int current_y, int n)
{
    double d = 0;
    double K = get_K(image, depth, Mask, current_x, current_y,n);
    if(K==0)
    {
        return 0;
    }

    int topleft_x = current_x - (n - 1) / 2, topleft_y = current_y - (n - 1) / 2;
    for (int x = topleft_x; x < topleft_x + n; x++)
    {
        if (x < 0)
            continue;
        else if (x >= depth.rows)
            break;
        for (int y = topleft_y; y < topleft_y + n; y++)
        {
            if (y < 0)
                continue;
            else if (y >= depth.cols)
                break;

            double s_w = space_weight(current_x, current_y, x, y);
            double v_w = value_weight((int)image.at<uchar>(current_x, current_y), (int)image.at<uchar>(x, y));
            double r = get_R(image,depth,Mask,x,y,n);
            int d_v = (int)depth.at<uchar>(x, y);

            d += s_w * v_w * r * d_v;
        }
    }

    return d / K;
}


