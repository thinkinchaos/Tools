#ifndef __DETECT_H_
#define __DETECT_H_

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class CDetect
{
public:
	CDetect() {};
	~CDetect() {};
	struct SObjsInfo
	{
		Rect RECT;
		Point BottomMiddle;
		Point Barycenter;
		Point TopMiddle;
		vector<Point> ObjContour;
		int RectArea;
		int already_save_flag;
		vector<Point> track;
		//bool x_change_flag;
		//bool y_change_flag;
	};
	vector<Point> track1;
	vector<Point> track2;
	vector<Point> track3;
	vector<Point> track4;
	vector<vector<Point>> tracks;
	//struct SObjsInfo2
	//{
	//	vector<Rect> RECTs;
	//	//Point BottomMiddles;
	//	Point Barycenter;
	//	Point TopMiddle;
	//	vector<Point> ObjContour;
	//	int RectArea;
	//	int already_save_flag;
	//	vector<Point> track;
	//	//bool x_change_flag;
	//	//bool y_change_flag;
	//};


	string VideoName = "rain.mov";
	string VideoRoot = "E:\\database\\video\\";

	bool bRunFlag = true;

	bool ChooseModel(int iModel);

	bool LoadAndWriteVideo(string filename1, string filename2);

	bool FindNearestPoint(vector<SObjsInfo> inputObjsInfo, Point inputPoint, Point &outputPoint);

	bool BehaviorJudgment(void);

	bool Motion_Target_Detection2(void);

	bool diff(Mat input1, Mat input2, Mat &output);

	bool RunFrameDifference(const bool &runFlag);
	bool RunGMM(const bool& runFlag);
	bool RunGMM2(const bool& runFlag);

	Mat equ, lap, log, gam, adg;
	bool AdaptGammaEnhance(cv::Mat Src, cv::Mat &Dst);
	bool HistogramEqualization(cv::Mat Src, cv::Mat &equ);
	bool LaplacianEnhance(cv::Mat Src, cv::Mat &Lap);
	bool LogarithmicTransformation(cv::Mat Src, cv::Mat &imageLog);
	bool GammaTransform(cv::Mat Src, cv::Mat &imageGamma);

	int numCount = 0;//已处理的帧计数
	int colorFlag;
	int can_match_flag;

	int frameH;		//获取帧高
	int frameW;		//获取帧宽
	int fps;                 //获取帧率
	int numFrames;   //获取整个帧数

	Mat frame;
	Mat drawing;
	VideoCapture capture;
	VideoWriter w_cap;

	vector<SObjsInfo> ObjsInfoNow;
	vector<SObjsInfo> ObjsInfoPre;
	vector<SObjsInfo> ObjsInfoPre2;

	vector<SObjsInfo> all_now;
	vector<SObjsInfo> all_pre;

	Mat first_pic;
	Mat last_pic;

	Mat inter_pic;
	Mat inter_pic2;
};

#endif