#include "detect.h"
Mat src;
bool ldown = false, lup = false;//点击行为
Point corner1, corner2;//开始点-结束点
Rect box;//ROIbox
static void mouse_callback(int event, int x, int y, int, void*)
{
	//点击鼠标左键
	if (event == EVENT_LBUTTONDOWN)
	{
		ldown = true;
		corner1.x = x;
		corner1.y = y;
		//cout << "corner1 recorded at" << corner1 << endl;
	}
	//释放鼠标左键
	if (event == EVENT_LBUTTONUP)
	{
		if (abs(x - corner1.x) > 20 && abs(y - corner1.y) > 20)
		{
			lup = true;
			corner2.x = x;
			corner2.y = y;
			//cout << "corner2 recorded at" << corner2 << endl;
		}
		else
		{
			ldown = false;
			//cout << "Please Select a bigger region! " << endl;
		}
	}

	if (ldown == true && lup == false)
	{
		Point pt;
		pt.x = x;
		pt.y = y;
		Mat local_img = src.clone();
		rectangle(local_img, corner1, pt, Scalar(0, 0, 255));
		cv::imshow("drawing", local_img);
	}
	//重新自定义
	if (ldown == true && lup == true)
	{
		box.width = abs(corner1.x - corner2.x);
		box.height = abs(corner1.y - corner2.y);
		box.x = min(corner1.x, corner2.x);
		box.y = min(corner1.y, corner2.y);

		ldown = false;
		lup = false;
	}
}

bool CDetect::AdaptGammaEnhance(cv::Mat Src, cv::Mat &Dst)
{
	cv::Mat Img = Src.clone();
	cv::cvtColor(Img, Img, COLOR_BGR2HSV);
	cv::Mat imageHSV[3];
	cv::split(Img, imageHSV);
	cv::Mat ImgV = imageHSV[2].clone();
	ImgV.convertTo(ImgV, CV_32FC1);
	cv::Mat imageVNomalize = cv::Mat::zeros(Src.size(), CV_32FC1);
	imageVNomalize = ImgV / 255.0;

	cv::Scalar     mean;
	cv::Scalar     dev;
	cv::meanStdDev(imageVNomalize, mean, dev);
	float       fMean = mean.val[0];
	float       detectev = dev.val[0];
	//std::cout << fMean << ",\t" << detectev << std::endl;

	float Gamma;
	float c;
	float k;
	int H;
	cv::Mat ImgOut = cv::Mat::zeros(Src.size(), CV_32FC1);
	H = (0.5 - fMean > 0) ? 1 : 0;

	if (detectev <= 0.0833) //低对比度图像 4detectev <= 1/12
	{
		Gamma = -(std::log(detectev) / std::log(2));
	}
	else
	{
		Gamma = exp((1 - (fMean + detectev)) / 2);
	}
	//std::cout << ",\t" << Gamma << std::endl;
	for (int i = 0; i < ImgOut.rows; i++)
	{
		for (int j = 0; j < ImgOut.cols; j++)
		{
			float IinGamma = pow(imageVNomalize.at<float>(i, j), Gamma);
			k = IinGamma + (1 - IinGamma)*pow(fMean, Gamma);
			c = 1.0 / (1 + H * (k - 1));
			ImgOut.at<float>(i, j) = c * pow(imageVNomalize.at<float>(i, j), Gamma)*255.0;
		}
	}
	//cv::normalize(ImgOut, ImgOut, 0, 255, CV_MINMAX);
	//转换成8bit图像显示  
	cv::convertScaleAbs(ImgOut, ImgOut);
	//ImgOut.convertTo(ImgOut, CV_8UC1);

	imageHSV[2] = ImgOut.clone();
	cv::merge(imageHSV, 3, Dst);
	cv::cvtColor(Dst, Dst, COLOR_HSV2BGR);
	imshow("AdaGamma", Dst);
	return true;
}
bool CDetect::HistogramEqualization(cv::Mat Src, cv::Mat &equ)
{
	Mat imageRGB[3];
	split(Src, imageRGB);
	for (int i = 0; i < 3; i++)
	{
		equalizeHist(imageRGB[i], imageRGB[i]);
	}
	merge(imageRGB, 3, equ);
	imshow("直方图均衡化图像增强效果", equ);
	return 0;
}
bool CDetect::LaplacianEnhance(cv::Mat Src, cv::Mat &Lap)
{
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, 5, 0, 0, -1, 0);
	filter2D(Src, Lap, CV_8UC3, kernel);
	imshow("拉普拉斯算子图像增强效果", Lap);
	return 0;
}
bool CDetect::LogarithmicTransformation(cv::Mat Src, cv::Mat &imageLog)
{
	imageLog = Mat::zeros(Src.size(), CV_32FC3);
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			imageLog.at<Vec3f>(i, j)[0] = std::log(1 + Src.at<Vec3b>(i, j)[0]);
			imageLog.at<Vec3f>(i, j)[1] = std::log(1 + Src.at<Vec3b>(i, j)[1]);
			imageLog.at<Vec3f>(i, j)[2] = std::log(1 + Src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageLog, imageLog, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageLog, imageLog);
	imshow("log", imageLog);
	return 0;
}
bool CDetect::GammaTransform(cv::Mat Src, cv::Mat &imageGamma)
{
	imageGamma = Mat::zeros(Src.size(), CV_32FC3);
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			imageGamma.at<Vec3f>(i, j)[0] = (Src.at<Vec3b>(i, j)[0])*(Src.at<Vec3b>(i, j)[0])*(Src.at<Vec3b>(i, j)[0]);
			imageGamma.at<Vec3f>(i, j)[1] = (Src.at<Vec3b>(i, j)[1])*(Src.at<Vec3b>(i, j)[1])*(Src.at<Vec3b>(i, j)[1]);
			imageGamma.at<Vec3f>(i, j)[2] = (Src.at<Vec3b>(i, j)[2])*(Src.at<Vec3b>(i, j)[2])*(Src.at<Vec3b>(i, j)[2]);
		}
	}
	//归一化到0~255  
	normalize(imageGamma, imageGamma, 0, 255, NORM_MINMAX);
	//转换成8bit图像显示  
	convertScaleAbs(imageGamma, imageGamma);
	imshow("伽马变换图像增强效果", imageGamma);
	return 0;
}

bool CDetect::LoadAndWriteVideo(string filename1, string filename2)//读和写视频
{
	capture.open(filename1);
	if (!capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return false;
	}

	frameH = capture.get(CAP_PROP_FRAME_HEIGHT);		//获取帧高
	frameW = capture.get(CAP_PROP_FRAME_WIDTH);		//获取帧宽
	fps = capture.get(CAP_PROP_FPS);                 //获取帧率
	numFrames = capture.get(CAP_PROP_FRAME_COUNT);   //获取整个帧数

	printf(" video's \n width = %d \n height = %d \n video's fps = %d \n nums = %d \n", frameW, frameH, fps, numFrames);

	w_cap.open(filename2, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameW, frameH));

	return true;
}

bool CDetect::FindNearestPoint(vector<SObjsInfo> inputObjsInfo, Point inputPoint, Point &outputPoint)
{
	int minDist = 60000;
	for (int i = 0; i < inputObjsInfo.size(); i++)
	{
		int disTmp = powf((inputObjsInfo[i].Barycenter.x - inputPoint.x), 2) + powf((inputObjsInfo[i].Barycenter.y - inputPoint.y), 2);
		if (disTmp < minDist)
		{
			minDist = disTmp;
		}
	}
	for (int i = 0; i < inputObjsInfo.size(); i++)
	{
		int disTmp = powf((inputObjsInfo[i].Barycenter.x - inputPoint.x), 2) + powf((inputObjsInfo[i].Barycenter.y - inputPoint.y), 2);
		if (minDist == disTmp)
		{
			outputPoint = Point(inputObjsInfo[i].Barycenter);
			break;
		}
	}
	return true;
}

bool CDetect::diff(Mat input1, Mat input2, Mat &output)
{
	cvtColor(input1, input1, COLOR_RGB2GRAY);
	GaussianBlur(input1, input1, Size(5, 5), 0);//高斯滤波的速度比中值滤波快
	cvtColor(input2, input2, COLOR_RGB2GRAY);
	GaussianBlur(input2, input2, Size(5, 5), 0);
	absdiff(input1, input2, output);//用帧差法求前景
	threshold(output, output, 70, 255, THRESH_BINARY);
	return true;
}

bool CDetect::Motion_Target_Detection2(void)
{
	Mat	diff1, diff2, diff3, thresh1, thresh2, thresh3, thing1, thing2;
	diff(first_pic, frame, diff1);
	adaptiveThreshold(diff1, thresh1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
	bitwise_not(thresh1, thresh1);
	diff(inter_pic, frame, diff2);
	adaptiveThreshold(diff2, thresh2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 15);
	diff(first_pic, inter_pic2, diff3);
	adaptiveThreshold(diff3, thresh3, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
	bitwise_not(thresh3, thresh3);
	multiply(thresh1, thresh2, thing1);
	multiply(thresh1, thresh3, thing2);
	Mat element = cv::getStructuringElement(MORPH_ELLIPSE, cv::Size(15, 25));
	Mat dilate,erode;
	cv::morphologyEx(diff1, dilate, MORPH_DILATE, element);
	cv::morphologyEx(dilate, erode, MORPH_ERODE, element);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(dilate, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	//get cur objs info
	ObjsInfoNow.clear();
	for (size_t i = 0; i < contours.size(); i++)
	{
		//cout << contourArea(contours[i]) << endl;
		int area = contourArea(contours[i]);
		if (area > 800)
		{
			Rect rect = boundingRect(contours[i]);
			Point center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
			SObjsInfo obj_tmp;
			obj_tmp.RECT = rect;
			obj_tmp.RectArea = area;
			obj_tmp.Barycenter = center;
			obj_tmp.ObjContour = contours[i];
			ObjsInfoNow.push_back(obj_tmp);
		}
	}

	if (numCount == 0)
	{
		ObjsInfoPre = ObjsInfoNow;	
		all_pre = ObjsInfoNow;
		all_now = ObjsInfoNow;
	}

	//match pre objs as pre2
	ObjsInfoPre2.clear();
	if (ObjsInfoNow.size() == ObjsInfoPre.size())
	{
		for (int i = 0; i < ObjsInfoNow.size(); i++)
		{
			Point NearestPointTmp;
			FindNearestPoint(ObjsInfoPre, ObjsInfoNow[i].Barycenter, NearestPointTmp);
			for (int j = 0; j < ObjsInfoPre.size(); j++)
			{
				if (NearestPointTmp == ObjsInfoPre[j].Barycenter)
				{
					SObjsInfo ContoursInfoTmp;
					ContoursInfoTmp.RECT = ObjsInfoPre[j].RECT;
					ContoursInfoTmp.RectArea = ObjsInfoPre[j].RectArea;
					ContoursInfoTmp.Barycenter = ObjsInfoPre[j].Barycenter;
					//ContoursInfoTmp.BottomMiddle = ObjsInfoPre[j].BottomMiddle;
					//ContoursInfoTmp.TopMiddle = ObjsInfoPre[j].TopMiddle;
					ContoursInfoTmp.ObjContour = ObjsInfoPre[j].ObjContour;
					ObjsInfoPre2.push_back(ContoursInfoTmp);
				}
			}
		}

		//for (int i = 0; i < ObjsInfoNow.size(); i++)
		//{
		//	int derviate_x, derviate_y;
		//	derviate_x = ObjsInfoNow[i].Barycenter.x - ObjsInfoPre2[i].Barycenter.x;
		//	derviate_y = ObjsInfoNow[i].Barycenter.y - ObjsInfoPre2[i].Barycenter.y;
		//	if (derviate_y > 0)
		//	{
		//		colorFlag = 0;
		//	}
		//	if (derviate_y < 0)
		//	{
		//		colorFlag = 1;
		//	}
		//	if (derviate_y < 1 && derviate_x < 1)//remain
		//	{
		//		colorFlag = 2;
		//	}
		//	if (colorFlag == 0)
		//	{
		//		putText(drawing, "in", Point(ObjsInfoNow[i].RECT.x, ObjsInfoNow[i].RECT.y - 1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
		//		rectangle(drawing, ObjsInfoNow[i].RECT, Scalar(0, 255, 0));
		//	}
		//	else if (colorFlag == 1)
		//	{
		//		putText(drawing, "out", Point(ObjsInfoNow[i].RECT.x, ObjsInfoNow[i].RECT.y - 1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
		//		rectangle(drawing, ObjsInfoNow[i].RECT, Scalar(0, 0, 255));
		//	}
		//	else if (colorFlag == 2)
		//	{
		//		putText(drawing, "remain", Point(ObjsInfoNow[i].RECT.x, ObjsInfoNow[i].RECT.y - 1), FONT_HERSHEY_PLAIN, 1, Scalar(255, 0, 0));
		//		rectangle(drawing, ObjsInfoNow[i].RECT, Scalar(255, 0, 0));
		//	}
		//	cv::imshow("drawing", drawing);
		//}
	}

	if (all_now.empty() == 1 && ObjsInfoNow.empty() == 0)
	{
		all_now = ObjsInfoNow;
	}

	//save new obj and get all objs num
	vector<SObjsInfo> new_tmp;
	new_tmp.clear();
	for (int j = 0; j < ObjsInfoNow.size(); j++)
	{
		//get min dist
		int minDist = 60000;
		for (int i = 0; i < all_now.size(); i++)
		{
			int disTmp = powf((all_now[i].Barycenter.x - ObjsInfoNow[j].Barycenter.x), 2) + powf((all_now[i].Barycenter.y - ObjsInfoNow[j].Barycenter.y), 2);
			if (disTmp < minDist)
			{
				minDist = disTmp;
			}
		}

		//得到，现在的点的，最近的已知的点
		Point NearestPointTmp;
		int id;
		for (int i = 0; i < all_now.size(); i++)
		{
			int disTmp = powf((all_now[i].Barycenter.x - ObjsInfoNow[j].Barycenter.x), 2) + powf((all_now[i].Barycenter.y - ObjsInfoNow[j].Barycenter.y), 2);
			if (minDist == disTmp)
			{
				NearestPointTmp = Point(all_now[i].Barycenter);
				id = i;
				break;
			}
		}

		int dx = abs(NearestPointTmp.x - ObjsInfoNow[j].Barycenter.x);
		int dy = abs(NearestPointTmp.y - ObjsInfoNow[j].Barycenter.y);
		//std::cout << "x y "<< dx << " " << dy << endl;
		if (dx > 100 && dy > 100)//new obj
		{
			new_tmp.push_back(ObjsInfoNow[j]);
			//cout << "save" << endl;
		}
		else
		{
			//replase
			//cout << "dd" << endl;
			all_now[id].RECT = ObjsInfoNow[j].RECT;
			all_now[id].Barycenter = ObjsInfoNow[j].Barycenter;

			all_now[id].track.push_back(ObjsInfoNow[j].Barycenter);
			continue;
		}
	}
	all_now.insert(all_now.end(), new_tmp.begin(), new_tmp.end());
	//cout << ObjsInfoNow.size() << " " << all_now.size() << endl;

	//for (int j = 0; j < ObjsInfoNow.size(); j++)
	//{
	//	track1.push_back(ObjsInfoNow[0]);
	//}

	for (int i = 0; i < all_now.size(); i++)
	{
		if(all_now.size()>3)
		cout << i <<" "<<all_now[i].track.size() << endl;
		rectangle(drawing, all_now[i].RECT, Scalar(0, 255, 0));
		//for (int j = 0; j < all_now[i].track.size()-1; j++)
		//{
		//	line(drawing, all_now[i].track[j], all_now[i].track[j+1],Scalar(255,0,0));
		//}
		imshow("drawing", drawing);
	}

	cv::waitKey(1);
	inter_pic = frame.clone();
	ObjsInfoPre = ObjsInfoNow;
	return true;
}

bool CDetect::RunFrameDifference(const bool &runFlag)
{
	while (runFlag)
	{
		if (numCount < numFrames - 5)//没到尾帧
		{
			capture >> frame;
			//capture >> frame;
			//capture >> frame;
			resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
			drawing = frame.clone();//所有信息的展示图
			if (numCount == 0)//读入第一帧时，第一帧作为背景
			{
				first_pic = frame.clone();//保存第一帧
				inter_pic = first_pic.clone();
				inter_pic2 = first_pic.clone();

				src = first_pic.clone();
				imshow("drawing", src);
				setMouseCallback("drawing", mouse_callback);
				cv::waitKey(0);
			}

			Motion_Target_Detection2();

			numCount = numCount + 1;
		}
		else//快到尾帧
		{
			//last_pic = frame.clone();
			//imshow("last", last_pic);
			//waitKey(0);
			//Legacy_Target_Detection();
			break;
		}
	}

	return true;
}

bool CDetect::RunGMM(const bool& runFlag)
{
	if(runFlag==1)
	{
		Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
		bgsubtractor->setHistory(20);
		bgsubtractor->setVarThreshold(100);

		VideoCapture cap("e:/database/rain.mov");
		if (!cap.isOpened())
		{
			cout << "video not exist!" << endl;
			return -1;
		}
		long FRAMECNT = cap.get(CAP_PROP_FRAME_COUNT);
		int frameW = cap.get(CAP_PROP_FRAME_WIDTH);
		int frameH = cap.get(CAP_PROP_FRAME_HEIGHT);
		int fps = cap.get(CAP_PROP_FPS);

		VideoWriter writer("e:/test_3.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
			Size(frameW / 2, frameH / 2), 1);

		Rect box;

		Rect obj_cur;
		Rect obj_pre;

		Mat frame, mask, mask2;
		int frame_count = 0;
		int intrance_count = 0;
		int show_text_count = 0;
		while (++frame_count < FRAMECNT - 10)
		{
			//intrance_count++;
			//if()

			cap >> frame;
			resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));




			bgsubtractor->apply(frame, mask, 0.01);

			medianBlur(mask, mask, 3);
			Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
			Mat element2 = getStructuringElement(MORPH_RECT, Size(9, 9));
			erode(mask, mask, element1);
			dilate(mask, mask, element1);


			box.x = frame.cols / 4;
			box.y = frame.rows / 4;
			box.width = frame.cols / 2;
			box.height = frame.rows / 2;
			rectangle(frame, box, Scalar(0, 255, 0), 2);

			Mat roi = Mat::zeros(frame.size(), CV_8UC1);
			roi(box).setTo(255);
			mask.copyTo(mask2, roi);
			//imshow("only_roi_mat", only_roi_frame);



			//maskCp = mask.clone();
			vector<vector<Point>> cnts;
			findContours(mask2, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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
					rectangle(frame, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);
					intrance_count++;
				}

				obj_pre = obj_cur;
			}

			if (intrance_count > 10)
			{
				putText(frame, "Alert!", Point(box.x + 2, box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
				show_text_count++;
				//waitKey(20);
				if (show_text_count > 10)
				{
					intrance_count = 0;
					show_text_count = 0;
				}
			}



			imshow("frame", frame);
			imshow("mask", mask);
			//writer.write(frame);
			waitKey(1);

		}
		
	}
	return true;
}

bool CDetect::ChooseModel(int iModel)
{

	if (iModel == 0) // 模式0 帧差法 检测视频
	{
		string in = CDetect::VideoRoot + CDetect::VideoName;
		string out = in + ".avi";
		cout << "input path: " << in << endl;
		cout << "output path: " << " " << out << endl;
		CDetect::LoadAndWriteVideo(in, out);
		CDetect::RunFrameDifference(bRunFlag);
	}
	else if (iModel == 1) // 模式1 GMM 检测视频
	{
		cout << "running model 1 : GMM for video";
		string in = CDetect::VideoRoot + CDetect::VideoName;
		string out = in + ".avi";
		cout << "input path: " << in << endl;
		cout << "output path: " << " " << out << endl;
		CDetect::LoadAndWriteVideo(in, out);

		CDetect::RunGMM(bRunFlag);
	}
	else if (iModel == 2) // 模式2 GMM 检测 摄像头
	{
		if (bRunFlag == 1)
		{
			Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
			bgsubtractor->setHistory(20);
			bgsubtractor->setVarThreshold(100);

			VideoCapture cap(0);

			//long FRAMECNT = cap.get(CAP_PROP_FRAME_COUNT);
			int frameW = cap.get(CAP_PROP_FRAME_WIDTH);
			int frameH = cap.get(CAP_PROP_FRAME_HEIGHT);
			//int fps = cap.get(CAP_PROP_FPS);

			Rect box;

			Rect obj_cur;
			Rect obj_pre;

			Mat frame, mask, mask2;
			int frame_count = 0;
			int intrance_count = 0;
			int show_text_count = 0;
			while (1)
			{
				//intrance_count++;
				//if()

				cap >> frame;
				resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));




				bgsubtractor->apply(frame, mask, 0.01);

				medianBlur(mask, mask, 3);
				Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
				Mat element2 = getStructuringElement(MORPH_RECT, Size(9, 9));
				erode(mask, mask, element1);
				dilate(mask, mask, element1);


				box.x = frame.cols / 4;
				box.y = frame.rows / 4;
				box.width = frame.cols / 2;
				box.height = frame.rows / 2;
				rectangle(frame, box, Scalar(0, 255, 0), 2);

				Mat roi = Mat::zeros(frame.size(), CV_8UC1);
				roi(box).setTo(255);
				mask.copyTo(mask2, roi);
				//imshow("only_roi_mat", only_roi_frame);



				//maskCp = mask.clone();
				vector<vector<Point>> cnts;
				findContours(mask2, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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
						rectangle(frame, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);
						intrance_count++;
					}

					obj_pre = obj_cur;
				}

				if (intrance_count > 10)
				{
					putText(frame, "Alert!", Point(box.x + 2, box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
					show_text_count++;
					//waitKey(20);
					if (show_text_count > 10)
					{
						intrance_count = 0;
						show_text_count = 0;
					}
				}



				imshow("frame", frame);
				imshow("mask", mask);
				//writer.write(frame);
				waitKey(1);

			}

		}
	}
	else if (iModel == 3) // 模式3 GMM 视频 完美版
	{
		std::cout << "Choose Method 3" << endl;
		
		CDetect::RunGMM2(bRunFlag);
	}
	//else if (iModel == 4) // 模式2 GMM检测视频
	//{
	//	CDetect::RunFrameDifference(bRunFlag);
	//}
	//else if (iModel == 5) // 模式3 GMM检测摄像头
	//{
	//	std::cout << "model choice error" << endl;
	//}
	else
	{
		//std::cout << "model choice error" << endl;
	}

	return true;
}

int get_threshold(Mat input, float coefficient)
{
	if (input.channels() != 1)
		return false;
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(input, minp, maxp);
	//cout << "Mat minv = " << minv << endl;
	//cout << "Mat maxv = " << maxv << endl;
	return (int)(maxv * coefficient);
}


bool CDetect::RunGMM2(const bool& runFlag)
{
	string in = CDetect::VideoRoot + CDetect::VideoName;
	string out = in + ".avi";
	cout << "input path: " << in << endl;
	cout << "output path: " << " " << out << endl;

	capture.open(in);
	if (!capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return false;
	}

	frameH = capture.get(CAP_PROP_FRAME_HEIGHT);		//获取帧高
	frameW = capture.get(CAP_PROP_FRAME_WIDTH);		//获取帧宽
	fps = capture.get(CAP_PROP_FPS);                 //获取帧率
	numFrames = capture.get(CAP_PROP_FRAME_COUNT);   //获取整个帧数

	printf(" video's \n width = %d \n height = %d \n video's fps = %d \n nums = %d \n", frameW, frameH, fps, numFrames);

	w_cap.open(out, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(frameW, frameH));

	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setHistory(20);
	bgsubtractor->setVarThreshold(100);

	Mat drawing;

	Rect obj_cur;
	Rect obj_pre;

	Mat frame, mask, mask2;
	int frame_count = 0;
	int intrance_count = 0;
	int show_text_count = 0;

	Mat retain_pre, retain_cur;

	Mat background;

	while (runFlag == true && frame_count < numFrames - 10)
	{
		capture >> frame;
		resize(frame, frame, Size(frameW / 2, frameH / 2));

		drawing = frame.clone();

		bgsubtractor->apply(frame, mask, 0.01);

		//retain_cur = frame.clone();
		//GaussianBlur(retain_cur, retain_cur, Size(5, 5), 2);

		if (frame_count == 0)
		{
			//retain_pre = retain_cur;
			background = frame.clone();


			src = frame.clone();
			putText(src, "Set ROI First", Point(2, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
			imshow("drawing", src);
			setMouseCallback("drawing", mouse_callback);
			cv::waitKey(0);
		}

		//if (frame_count % 50 == 0 && frame_count != 0)
		//{
		//	Mat retain;

		//	absdiff(retain_pre, retain_cur, retain);//用帧差法求前景
		//	cvtColor(retain, retain, COLOR_RGB2GRAY);
		//	threshold(retain, retain, get_threshold(retain, 0.95), 255, THRESH_BINARY);

		//	medianBlur(retain, retain, 3);
		//	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
		//	erode(retain, retain, element1);
		//	dilate(retain, retain, element1);

		//	vector<vector<Point>> cnts1;
		//	findContours(retain, cnts1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//	vector<Point> maxCnt1;
		//	if (cnts1.size() > 0)
		//	{
		//		for (int i = 0; i < cnts1.size(); ++i)
		//		{
		//			maxCnt1 = maxCnt1.size() > cnts1[i].size() ? maxCnt1 : cnts1[i];
		//		}
		//	}
		//	Rect retain_box = boundingRect(maxCnt1);

		//	rectangle(drawing, retain_box, Scalar(255, 0, 0), 2);
		//	putText(drawing, "Remnant", Point(retain_box.x + 2, retain_box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

		//	retain_pre = retain_cur;
		//}


		rectangle(drawing, box, Scalar(0, 255, 0), 2);

		medianBlur(mask, mask, 3);
		Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
		erode(mask, mask, element1);
		dilate(mask, mask, element1);

		Mat roi = Mat::zeros(frame.size(), CV_8UC1);
		roi(box).setTo(255);
		mask.copyTo(mask2, roi);

		vector<vector<Point>> cnts;
		findContours(mask2, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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
			//cout << intersect.area() << endl;
			if (intersect.area() > 30)
			{
				rectangle(drawing, obj_cur.tl(), obj_cur.br(), Scalar(0, 0, 255), 2);
				intrance_count++;
			}

			obj_pre = obj_cur;
		}

		if (intrance_count > 10)
		{
			cout << "Something intrance!" << endl;
			putText(drawing, "Alert!", Point(box.x + 2, box.y + 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
			show_text_count++;
			//waitKey(20);
			if (show_text_count > 10)
			{
				intrance_count = 0;
				show_text_count = 0;
			}
		}



		imshow("drawing", drawing);
		//imshow("mask", mask);
		//writer.write(frame);
		waitKey(1);

		frame_count++;
	}
	return true;
}