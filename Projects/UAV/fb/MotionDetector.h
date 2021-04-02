/*****************************************************************************
* @FileName MotionDetector.h
* @Author: FengBo
* @Email:fb_941219@163.com
* @CreatTime: 2019/9/19 
* @Descriptions:
* @Version: ver 1.0
* @Copyright(c) 2019 All Rights Reserved.
*****************************************************************************/
#ifndef MOTIONDETECTOR_H_
#define MOTIONDETECTOR_H_

#include "opencv2/opencv.hpp"
#include "opencv2/video/background_segm.hpp"
#include <stdlib.h>
//#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iostream>
#include <cmath>
#include <ctime>
#include <windows.h>
#include <Mmsystem.h>
#include "Timer.h"
#include <thread>
#include "thread.h"
#include <chrono>         // std::chrono::system_clock
#pragma comment(lib,"winmm.lib")

#include"omp.h"



using namespace std;
using namespace cv;

class MotionDetector
{
public:
	MotionDetector();
	~MotionDetector();

public:
	void GmmDetector();
	static void MomouseCallback(int event, int x, int y, int, void*);
	void Warn();

	Timer timer;
	mutex mu; 
	   	 
	
private:


	//线程暂停计时
	void  pausable();
	//路径
	string name = "rain.MOV";
	string root = "E:\\database\\video\\";
	#define SOUND_PATH "E:\\database\\video\\sound.wav"
	//resize
	int resizeN = 2;

	//drawing
	static Point point_start;
	static Point point_end;//开始点-结束点
	static Mat drawing;
	static Rect roi_box;//ROIroi_box
	static bool left_down;
	static bool left_up;//点击行为

	//detector
	long FRAMECNT;
	int frameW;
	int frameH;
	int fps;


	Rect obj_cur, obj_pre;
	int frame_count = 0;
	int intrance_count = 0;
	int show_text_count = 0;
	

	Mat gmm_result;
	Mat frame_roi;
	Mat processed_result;


	//第一帧报警消除标志
	bool SoundFlag = false;
	bool FirstTime = true;
	bool WarnFlag = false;

	
	//clock_t startTime, endTime;
	int numProcs = omp_get_num_procs();
	   
};


#endif


