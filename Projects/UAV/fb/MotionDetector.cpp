/*****************************************************************************
* @FileName MotionDetector.cpp
* @Author: FengBo
* @Email:fb_941219@163.com
* @CreatTime: 2019/9/19 
* @Descriptions:
* @Version: ver 1.0
* @Copyright(c) 2019 All Rights Reserved.
*****************************************************************************/
#include "MotionDetector.h"

bool MotionDetector::left_down = false;
bool MotionDetector::left_up = false;//点击行为
Point MotionDetector::point_start = Point();
Point MotionDetector::point_end = Point();//开始点-结束点
Mat MotionDetector::drawing = Mat();
Rect MotionDetector::roi_box = Rect();//ROIroi_box

MotionDetector::MotionDetector()
{
}

MotionDetector::~MotionDetector()
{
}


void MotionDetector::MomouseCallback(int event, int x, int y, int, void*)
{
	//点击鼠标左键
	if (event == EVENT_LBUTTONDOWN)
	{
		left_down = true;
		point_start.x = x;
		point_start.y = y;
		//cout << "point_start recorded at" << point_start << endl;
	}
	//释放鼠标左键
	if (event == EVENT_LBUTTONUP)
	{
		if (abs(x - point_start.x) > 20 && abs(y - point_start.y) > 20)
		{
			left_up = true;
			point_end.x = x;
			point_end.y = y;
			//cout << "point_end recorded at" << point_end << endl;
		}
		else
		{
			left_down = false;
			//cout << "Please Select a bigger region! " << endl;
		}
	}

	if (left_down == true && left_up == false)
	{
		Point pt;
		pt.x = x;
		pt.y = y;
		Mat tmp = drawing.clone();
		rectangle(tmp, point_start, pt, Scalar(0, 0, 255));
		cv::imshow("drawing", tmp);
	}
	//重新自定义
	if (left_down == true && left_up == true)
	{
		roi_box.width = abs(point_start.x - point_end.x);
		roi_box.height = abs(point_start.y - point_end.y);
		roi_box.x = min(point_start.x, point_end.x);
		roi_box.y = min(point_start.y, point_end.y);

		left_down = false;
		left_up = false;
	}
}


void MotionDetector::GmmDetector()
{
	
	
	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setHistory(20);
	bgsubtractor->setVarThreshold(100);

	VideoCapture cap(root + name);
	if (!cap.isOpened())
	{
		cout << "video not exist!" << endl;
		//return -1;
		exit(0);
	}

	FRAMECNT = cap.get(CAP_PROP_FRAME_COUNT);
	frameW = cap.get(CAP_PROP_FRAME_WIDTH);
	frameH = cap.get(CAP_PROP_FRAME_HEIGHT);
	fps = cap.get(CAP_PROP_FPS);

	VideoWriter writer(root + name + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
		Size(frameW / resizeN, frameH / resizeN), 1);


	while (++frame_count < FRAMECNT - 10)
	{
		Mat frame;
		cap >> frame;
	
		
		resize(frame, frame, Size(frameW / resizeN, frameH / resizeN));

		Mat roi = Mat::zeros(frame.size(), CV_8UC1);
		roi(roi_box).setTo(255);
		drawing = frame.clone();

		if (frame_count == 1)
		{
			imshow("drawing", drawing);
			setMouseCallback("drawing",MomouseCallback);
			cv::waitKey(0);
		}
		
		if (roi_box.width > 2)
		{
			rectangle(drawing, roi_box, Scalar(0, 255, 0), 2);
			frame.copyTo(frame_roi, roi);

			bgsubtractor->apply(frame_roi, gmm_result, 0.01);

		}
		else
		{
			bgsubtractor->apply(frame, gmm_result, 0.01);
			processed_result = gmm_result.clone();
		}
		#pragma omp parallel num_threads(2*numProcs-1)
		{
			//int num = omp_get_num_procs();
			//cout << num << endl;
			medianBlur(gmm_result, processed_result, 3);
			Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
			//Mat element2 = getStructuringElement(MORPH_RECT, Size(9, 9));
			erode(processed_result, processed_result, element1);
			//dilate(processed_result, processed_result, element1);
			medianBlur(processed_result, processed_result, 3);
		}
		imshow("processed_result", processed_result);


		vector<vector<Point>> cnts;
		findContours(processed_result, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		vector<Point> maxCnt;
		
		// 为了屏蔽建筑物移动
		if (cnts.size() > 0 && cnts.size() < 5)
		{
			for (int i = 0; i < cnts.size(); ++i)
			{
				maxCnt = maxCnt.size() > cnts[i].size() ? maxCnt : cnts[i];
			}

			Rect obj_cur = boundingRect(maxCnt);

			// 为了屏蔽雨水等噪声
			Rect intersect = obj_cur & obj_pre;
			cout << intersect.area() << endl;
			if (intersect.area() > 30)
			{
				rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);
				intrance_count++;
			}

			obj_pre = obj_cur;
		}
		//cout << "intrance_count:" << intrance_count << endl;
		if (intrance_count>10 && FirstTime)//第一帧不知道为何要响
		{
			FirstTime = false;
			SoundFlag = true;
			intrance_count = 0;
		}
		if (intrance_count > 10 && SoundFlag)
		{			
			
			//timer.registerHandler(Warn,NULL);
			//timer.setInterval(2000);
			//timer.Start();

			//std::thread t1(bind(&MotionDetector::Warn, this));
			//t1.join();
			//报警点
			//mu.lock(); //同步数据锁
			WarnFlag = true;
		    //mu.unlock();  //解除锁定
			
			putText(drawing, "Warnning!", Point(roi_box.x + 2, roi_box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);//光（屏幕显示）
			show_text_count++;
			waitKey(20);
			if (show_text_count > 5)
			{
				intrance_count = 0;
				show_text_count = 0;
			}
		}

		//writer.write(frame);

		imshow("drawing", drawing);
		waitKey(1);
		//}
	}
	//return true;
};

 void MotionDetector::Warn()
{
	 while (1)
	 {
		 if (WarnFlag)
		 {
			 //mu.lock(); //同步数据锁
			//PlaySound(NULL, NULL, SND_FILENAME | SND_ASYNC);//声
			 PlaySound(TEXT(SOUND_PATH), NULL, SND_FILENAME | SND_ASYNC);//声
			//putText(drawing, "Warnning!", Point(roi_box.x + 2, roi_box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);//光（屏幕显示）
			 std::this_thread::sleep_for(std::chrono::seconds(3));    //暂停30秒
			 PlaySound(NULL, NULL, SND_FILENAME | SND_ASYNC);//声音停止
			// putText(drawing, "Warnning!", Point(roi_box.x + 2, roi_box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);//光（屏幕显示）
			 //waitKey(2000);
			 WarnFlag = false;
			 //mu.unlock();  //解除锁定
		 }
		 //mu.lock(); //同步数据锁
		 //Sleep(100);
		 //mu.unlock();  //解除锁定
		 //waitKey(100);
		 //sndPlaySound(TEXT(SOUND_PATH), SND_ASYNC);
		 //TODO： WarnFlag变量互锁,声光分离！
	 }
}


 void  MotionDetector::pausable() {
	 //// sleep 500毫秒
	 //std::this_thread::sleep_for(milliseconds(500));
	 //// sleep 到指定时间点
	 //std::this_thread::sleep_until(system_clock::now() + milliseconds(500));
 }
