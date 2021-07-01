#include <iostream>

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("2.jpg");
	imshow("1.ȱ�ݲ���", src);

	Mat bil;
	bilateralFilter(src, bil, 25, 25 * 2, 25 / 2);//˫���˲�
	imshow("2.˫���˲�", bil);

	Mat gray;
	cvtColor(bil, gray, COLOR_BGR2GRAY);
	
	//��һ�ζ�ֵ��
	for (int i = 0; i < gray.cols; i++)
	{
		int min = 256;
		for (int j = 0; j < gray.rows; j++)//�Ҹ��е���С�Ҷ�ֵ
		{
			if (gray.at<uchar>(j, i) < min)//ע�⣡������ j i
			{
				min = gray.at<uchar>(j, i);
			}
		}

		for (int j = 0; j < gray.rows; j++)//�Ը��ж�ֵ��
		{
			if (gray.at<uchar>(j, i) == min)//��ֵΪ��Сֵ���������Ա�֤�����ж����ѷ�
			{
				gray.at<uchar>(j, i) = 255;
			}
			else
			{
				gray.at<uchar>(j, i) = 0;
			}
		}
	}
	imshow("3.��һ�ζ�ֵ��", gray);

	//�˳�������
	Mat bw = Mat::zeros(gray.size(), gray.type());//����һ����ԭͼ�ȴ�Ŀհ׵�ͼ
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
				roi.copyTo(bw(rect));//���������ڲ�ֻ��һ�����������ݣ�������ô�������Ч���ڣ��Ѹô��ڵ����ݸ��Ƶ��հ�ͼ��Ӧ������
			}
		}
	}
	imshow("4.�˳�����ֵ", bw);

	Mat gau;
    GaussianBlur(bw, gau, Size(9,9), 3);//��ģ���˹ģ��
	imshow("5.��˹ģ��", gau);

	threshold(gau, gau, 0, 255, THRESH_BINARY);//��ֵ������ֵ�� 0
	imshow("6.�ڶ��ζ�ֵ��", gau);

	Mat element1 = getStructuringElement(MORPH_RECT, Size(7, 7));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat m;
	erode(gau, m, element1);//��ģ�帯ʴ����ϸ��
	dilate(m, m, element2);//Сģ�����ͣ���ƽ��
	erode(m, m, element2);//Сģ�帯ʴ���ٴ�ϸ��
	imshow("7.��̬ѧ����", m);

	//�ҳ��������Ҫ��������������Ӧ���Ǹ�˹ģ��9*9�ļ���������ȡ2����Լ����200.
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
	//��������
	for (int i = 0; i < contours2.size(); i++)
	{
		cout << contourArea(contours2[i]) << endl;
		drawContours(src, contours2, -1, Scalar(0,0,255));
	}

	imshow("8.ȥ��С����������", src);
	waitKey();

	return 0;
};
