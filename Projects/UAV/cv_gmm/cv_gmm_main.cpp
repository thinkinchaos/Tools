#include <iostream>

#include "opencv2/opencv.hpp"

#include "opencv2/video/background_segm.hpp"

using namespace std;
using namespace cv;

//·��
string name = "rain.MOV";																			//��Ƶ����
string root = "E:\\database\\video\\";																//��ƵĿ¼
string videoStreamAddress = "rtsp://admin:test123456@192.168.10.65:554/MPEG-4/ch1/main/av_stream";	//��Ƶ����ַ

//������Ʋ���
int		videoModel = 0;				//��ȡģʽ��0--��Ƶ 1--RTSP  2--USB����ͷ
int		USB_CAMERA_PORT = 0;		//USB����ͷ�Ķ˿ں�
int		fps = 60;					//�洢��Ƶ��֡��
bool	saveFlag = 1;				//�Ƿ񱣴���
bool	runFlag = 1;				//���ⲿ�ж�
char	BREAK_KEY = 'q';			//�˳�����

//���Ч������
int		resizeN = 2;				//��С����ı���
bool	set_roi_flag = 0;			//�Ƿ�����ROI����
int		MAX_CONTOURS_NUM = 10;		//��֡��������������
int		MIN_INTERSECTION_AREA = 20;	//�ж�ΪĿ�����Сǰ��֡�ཻ���
int		INTRANCE_COUNT = 20;		//���ֶ���֡ʱ����
int		MIN_OBJ_AREA = 35;			//�ж�ΪĿ�����С���

//ȫ�ֱ���
Mat drawing;	//��ʾͼ
bool left_down = false, left_up = false;	//�����Ϊ
Point point_start, point_end;	//��ʼ��-������
Rect roi_box;	//ROI��
static void mouse_callback(int event, int x, int y, int, void*)//���ص�����
{
	//���������
	if (event == EVENT_LBUTTONDOWN)
	{
		left_down = true;
		point_start.x = x;
		point_start.y = y;
		//cout << "point_start recorded at" << point_start << endl;
	}
	//�ͷ�������
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
	//�����Զ���
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

	//������Ƶ����ģʽ
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

	//����GMM�Ĳ���
	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setHistory(20);
	bgsubtractor->setVarThreshold(100);

	Rect obj_cur, obj_pre;
	
	int intrance_count = 0;
	int show_text_count = 0;
	int frame_count = 0;

	while (1)
	{
		//��ֹ֡��ͳ��ֵ���
		frame_count++;
		if (frame_count > 30000)
			frame_count = 2;

		//������ⲿ��ֹ��������ֹ
		if (!runFlag)
		{
			writer.release();//�ͷ�д����
			break;
		}	
	
		Mat frame;
		cap.read(frame);
		if (frame.empty())//������Ƶ�����ˣ������ֹ
		{
			writer.release();//�ͷ�д����
			break;
		}

		int frameW = frame.cols;
		int frameH = frame.rows;
		int newW = frameW / resizeN;
		int newH = frameH / resizeN;
		resize(frame, frame, Size(newW, newH));

		drawing = frame.clone();

		if (frame_count == 1)//�ڵ�һ֡����ROI����,��������Ƶ�������
		{
			if (set_roi_flag == 1)
			{
				imshow("drawing", drawing);
				setMouseCallback("drawing", mouse_callback);
				cv::waitKey(0);
			}

			if (saveFlag == 1)//������Ƶ����Ĳ���
			{
				writer.open(root + name + ".avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
					Size(newW, newH), 1);
			}
		}

		Mat roi_frame = Mat::zeros(drawing.size(), drawing.type());
		if (roi_box.width > 2 && set_roi_flag)//���ROI����������Ч��������ֻѵ��������
		{
			rectangle(drawing, roi_box, Scalar(255, 0, 0), 2);

			Mat roi = frame(roi_box);//ROI������һС��
			roi.copyTo(roi_frame(roi_box));
		}
		else//���ROI����������Ч����ѵ������ͼ��
		{
			roi_frame = frame.clone();
		}
		
		Mat gmm_result;//GMM�ļ����
		cvtColor(roi_frame, roi_frame, COLOR_BGR2GRAY);
		bgsubtractor->apply(roi_frame, gmm_result, 0.01);
		imshow("gmm", gmm_result);

		Mat processed_result;//ͼ�����Ľ��
		medianBlur(gmm_result, processed_result, 3);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 9));
		erode(processed_result, processed_result, element1);
		dilate(processed_result, processed_result, element2);

		vector<vector<Point>> contours;
		findContours(processed_result, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		vector<Point> maxCnt;
		
		// Ϊ�����ν������ƶ�������֡��Ŀ�꣬��Ŀ�������࣬�������֡����Ϊ���������źܴ����Ч֡��
		if (contours.size() > 0 && contours.size() < MAX_CONTOURS_NUM)
		{
			for (int i = 0; i < contours.size(); ++i)
			{
				maxCnt = maxCnt.size() > contours[i].size() ? maxCnt : contours[i];
			}
			//�����������Ӿ�����Ϊ��ǰ֡��ǰ��Ŀ�ꡣȱ�㣺ÿ֡ͼ��ֻ�ܼ���һ��Ŀ�ꡣ
			obj_cur = boundingRect(maxCnt);

			// Ϊ��������ˮ����������ǰ��֮֡������������СС����ֵ���������֡(��Ϊ��ǰ��Ŀ��������ֵ)
			Rect intersect = obj_cur & obj_pre;
			//cout << intersect.area() << endl;
			if (intersect.area() > MIN_INTERSECTION_AREA && obj_cur.area() > MIN_OBJ_AREA)
			{
				rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 255, 0), 2);//��ʱ���
				intrance_count++;//ͳ�Ƹ�Ŀ����ֵĴ���
			}
		}

		if (intrance_count > INTRANCE_COUNT)//�����Ŀ����10֡�ж������ˣ���������
		{
			rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);//��ΪΣ�գ���ɫ���
			putText(drawing, "Alert", Point(obj_cur.tl().x, obj_cur.tl().y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);
			
			show_text_count++;//��ʾ�ı�����10֡
			if (show_text_count > 10)
			{
				intrance_count = 0;
				show_text_count = 0;
			}
		}

		obj_pre = obj_cur;//������ϣ�ǰ��Ŀ�����

		imshow("drawing", drawing);
		if (char(waitKey(1)) == BREAK_KEY)
		{
			writer.release();
			break;
		}

		if (saveFlag == 1)//���Ҫ��������
		{
			writer.write(drawing);
		}
	}
	return 0;
};
