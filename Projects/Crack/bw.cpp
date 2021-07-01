#include <iostream>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("2.jpg");
	imshow("1.缺陷部分", src);

	Mat bil;
	bilateralFilter(src, bil, 25, 25 * 2, 25 / 2);//双边滤波
	imshow("2.双边滤波", bil);

	Mat gray;
	cvtColor(bil, gray, COLOR_BGR2GRAY);
	
	//第一次二值化
	for (int i = 0; i < gray.cols; i++)
	{
		int min = 256;
		for (int j = 0; j < gray.rows; j++)//找该列的最小灰度值
		{
			if (gray.at<uchar>(j, i) < min)//注意！！！是 j i
			{
				min = gray.at<uchar>(j, i);
			}
		}

		for (int j = 0; j < gray.rows; j++)//对该列二值化
		{
			if (gray.at<uchar>(j, i) == min)//阈值为最小值附近，可以保证该列有多条裂缝
			{
				gray.at<uchar>(j, i) = 255;
			}
			else
			{
				gray.at<uchar>(j, i) = 0;
			}
		}
	}
	imshow("3.第一次二值化", gray);

	//滤除白噪声
	Mat bw = Mat::zeros(gray.size(), gray.type());//设置一个和原图等大的空白的图
	int blocksize = 3;
	for (int i = 1; i < gray.cols - blocksize +1; i++)
	{
		for (int j = 1; j < gray.rows - blocksize + 1; j++)
		{
			Rect rect(i, j, blocksize,blocksize);
			Mat roi = gray(rect);
			Scalar s = sum(roi);
			if (s[0] > 255)
			{
				roi.copyTo(bw(rect));//如果这个窗口不只有一个像素有数据，则表明该窗口是有效窗口，把该窗口的数据复制到空白图对应的区域
			}
		}
	}
	imshow("4.滤除奇异值", bw);

	Mat gau;
    GaussianBlur(bw, gau, Size(9,9), 3);//大模板高斯模糊
	imshow("5.高斯模糊", gau);

	threshold(gau, gau, 0, 255, THRESH_BINARY);//二值化，阈值是 0
	imshow("6.第二次二值化", gau);

	Mat element1 = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat m;
	erode(gau, m, element1);//大模板腐蚀，以细化
	dilate(m, m, element2);//小模板膨胀，以平滑
	erode(m, m, element2);//小模板腐蚀，再次细化
	imshow("7.形态学处理", m);

	//找出满足面积要求的轮廓。该面积应该是高斯模板9*9的几倍。这里取2倍，约等于200.
	vector<vector<Point>> contours;
	cv::findContours(m, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> contours2;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 200)
		{
			contours2.push_back(contours[i]);
		}
	}
	//画出轮廓
	for (int i = 0; i < contours2.size(); i++)
	{
		cout << contourArea(contours2[i]) << endl;
		drawContours(src, contours2, -1, Scalar(0,0,255));
	}

	imshow("8.去除小面积后的轮廓", src);
	waitKey();

	return 0;
};
