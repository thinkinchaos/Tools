#include <iostream>
#include <vector>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>



using namespace std;
using namespace cv;
using namespace cv::dnn;


// Initialize the parameters
float confThreshold = 0.2; // Confidence threshold
float nmsThreshold = 0.4;// Non-maximum suppression threshold
int inpWidth = 416;// Width of network's input image
int inpHeight = 416;// Height of network's input image
vector<string> classes;


Scalar colorTab[] = {
	Scalar(169,200,200),
	Scalar(53,57,130),
	Scalar(83,156,222),
	Scalar(52,116,64),
	Scalar(154,157,252),
	Scalar(155,175,131),
	Scalar(8,131,229),
	Scalar(18,87,220),
};

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), colorTab[classId], 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		//label = classes[classId] + ":" + label;
		label = classes[classId];
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	// cout << left<< top << endl;
	rectangle(frame, Point(left-1, top - labelSize.height), Point(left + labelSize.width, top), colorTab[classId], -1);
	putText(frame, label, Point(left, top - 2), FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1);

}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);

	}
}

void main()
{
	Size size = Size(640, 480);
	VideoWriter writer;
	int code = writer.fourcc('M','J','P','G');
	writer.open("1295_dst.avi", code, 25, size, true);


	string classesFile = "1295.names";
	String modelConfiguration = "1295.cfg";
	String modelWeights = "1295_final.weights";

	//string classesFile = "nuclear.names";
	//String modelConfiguration = "nuclear_voc_yolov3.cfg";
	//String modelWeights = "nuclear_voc_yolov3_last.weights";

	//string classesFile = "voc.names";
	//String modelConfiguration = "yolo-fastest.cfg";
	//String modelWeights = "yolo-fastest.weights";

	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);
	//for (int i = 0; i < classes.size(); i++)
	//{
	//	cout << classes[i] << endl;
	//}


	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	cv::VideoCapture cap("1295.avi");
	cv::Mat frame, blob;
	while (cap.read(frame))
	{
		//cap >> frame;

		cv::Mat sml;
		cv::resize(frame, sml, cv::Size(0, 0), 3.0 / 2, 3 / 2.0);
		blobFromImage(sml, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the
		// overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));

		// Write the frame with the detection boxes	
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);

		//cv::imshow("detectedFrame", detectedFrame);
		//cv::waitKey(1);
		writer.write(detectedFrame);
	}
}
