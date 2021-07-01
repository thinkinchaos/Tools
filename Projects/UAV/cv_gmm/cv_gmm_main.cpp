#include <iostream>

#include "opencv2/opencv.hpp"

#include "opencv2/video/background_segm.hpp"

using namespace std;
using namespace cv;

//路径
string name = "rain.MOV";																			//视频名称
string root = "E:\\database\\video\\";																//视频目录
string videoStreamAddress = "rtsp://admin:test123456@192.168.10.65:554/MPEG-4/ch1/main/av_stream";	//视频流地址

//程序控制参数
int		videoModel = 0;				//读取模式：0--视频 1--RTSP  2--USB摄像头
int		USB_CAMERA_PORT = 0;		//USB摄像头的端口号
int		fps = 60;					//存储视频的帧率
bool	saveFlag = 1;				//是否保存结果
bool	runFlag = 1;				//无外部中断
char	BREAK_KEY = 'q';			//退出按键

//检测效果参数
int		resizeN = 2;				//缩小处理的倍数
bool	set_roi_flag = 0;			//是否设置ROI区域
int		MAX_CONTOURS_NUM = 10;		//该帧允许的最大轮廓数
int		MIN_INTERSECTION_AREA = 20;	//判定为目标的最小前后帧相交面积
int		INTRANCE_COUNT = 20;		//入侵多少帧时报警
int		MIN_OBJ_AREA = 35;			//判定为目标的最小面积

//全局变量
Mat drawing;	//显示图
bool left_down = false, left_up = false;	//点击行为
Point point_start, point_end;	//开始点-结束点
Rect roi_box;	//ROI框
static void mouse_callback(int event, int x, int y, int, void*)//鼠标回调函数
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

int main()
{
	VideoCapture cap;
	VideoWriter writer;

	//设置视频输入模式
	if (videoModel == 0)//video
	{	
		if (!cap.open(root + name))
		{
			return -1;
		}
	}
	else if (videoModel == 1)//rtsp
	{
		if (!cap.open(videoStreamAddress))
		{
			return -1;
		}
	}
	else if (videoModel == 2)//usb camera
	{
		if (!cap.open(USB_CAMERA_PORT))
		{
			return -1;
		}
	}

	//设置GMM的参数
	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setHistory(20);
	bgsubtractor->setVarThreshold(100);

	Rect obj_cur, obj_pre;
	
	int intrance_count = 0;
	int show_text_count = 0;
	int frame_count = 0;

	while (1)
	{
		//防止帧数统计值溢出
		frame_count++;
		if (frame_count > 30000)
			frame_count = 2;

		//如果有外部中止命令，检测中止
		if (!runFlag)
		{
			writer.release();//释放写数据
			break;
		}	
	
		Mat frame;
		cap.read(frame);
		if (frame.empty())//若无视频输入了，检测中止
		{
			writer.release();//释放写数据
			break;
		}

		int frameW = frame.cols;
		int frameH = frame.rows;
		int newW = frameW / resizeN;
		int newH = frameH / resizeN;
		resize(frame, frame, Size(newW, newH));

		drawing = frame.clone();

		if (frame_count == 1)//在第一帧设置ROI区域,并设置视频保存参数
		{
			if (set_roi_flag == 1)
			{
				imshow("drawing", drawing);
				setMouseCallback("drawing", mouse_callback);
				cv::waitKey(0);
			}

			if (saveFlag == 1)//设置视频保存的参数
			{
				writer.open(root + name + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
					Size(newW, newH), 1);
			}
		}

		Mat roi_frame = Mat::zeros(drawing.size(), drawing.type());
		if (roi_box.width > 2 && set_roi_flag)//如果ROI区域设置有效，则框出，只训练该区域
		{
			rectangle(drawing, roi_box, Scalar(255, 0, 0), 2);

			Mat roi = frame(roi_box);//ROI区域那一小块
			roi.copyTo(roi_frame(roi_box));
		}
		else//如果ROI区域设置无效，则训练整个图像
		{
			roi_frame = frame.clone();
		}
		
		Mat gmm_result;//GMM的检测结果
		cvtColor(roi_frame, roi_frame, COLOR_BGR2GRAY);
		bgsubtractor->apply(roi_frame, gmm_result, 0.01);
		imshow("gmm", gmm_result);

		Mat processed_result;//图像处理后的结果
		medianBlur(gmm_result, processed_result, 3);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 9));
		erode(processed_result, processed_result, element1);
		dilate(processed_result, processed_result, element2);

		vector<vector<Point>> contours;
		findContours(processed_result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		vector<Point> maxCnt;
		
		// 为了屏蔽建筑物移动。若该帧无目标，或目标数过多，不处理该帧（认为是噪声干扰很大的无效帧）
		if (contours.size() > 0 && contours.size() < MAX_CONTOURS_NUM)
		{
			for (int i = 0; i < contours.size(); ++i)
			{
				maxCnt = maxCnt.size() > contours[i].size() ? maxCnt : contours[i];
			}
			//最大轮廓的外接矩形作为当前帧的前景目标。缺点：每帧图像只能检测出一个目标。
			obj_cur = boundingRect(maxCnt);

			// 为了屏蔽雨水等噪声。若前后帧之间的面积交集大小小于阈值，不处理该帧(认为该前景目标是奇异值)
			Rect intersect = obj_cur & obj_pre;
			//cout << intersect.area() << endl;
			if (intersect.area() > MIN_INTERSECTION_AREA && obj_cur.area() > MIN_OBJ_AREA)
			{
				rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 255, 0), 2);//及时框出
				intrance_count++;//统计该目标出现的次数
			}
		}

		if (intrance_count > INTRANCE_COUNT)//如果该目标在10帧中都出现了，发出警报
		{
			rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);//视为危险，红色框出
			putText(drawing, "Alert", Point(obj_cur.tl().x, obj_cur.tl().y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);
			
			show_text_count++;//显示文本持续10帧
			if (show_text_count > 10)
			{
				intrance_count = 0;
				show_text_count = 0;
			}
		}

		obj_pre = obj_cur;//处理完毕，前后目标更替

		imshow("drawing", drawing);
		if (char(waitKey(1)) == BREAK_KEY)
		{
			writer.release();
			break;
		}

		if (saveFlag == 1)//如果要保存检测结果
		{
			writer.write(drawing);
		}
	}
	return 0;
};
