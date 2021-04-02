/*****************************************************************************
* @FileName main.cpp
* @Author: FengBo
* @Email:fb_941219@163.com
* @CreatTime: 2019/9/19 
* @Descriptions:
* @Version: ver 1.0
* @Copyright(c) 2019 All Rights Reserved.
*****************************************************************************/
#include "MotionDetector.h"

int main()
{

	//union
	//{
	//	void *(*trfunc)(void *);
	//	void (MotionDetector::*memfunc)();
	//} func;

	//func.memfunc = &MotionDetector::GmmDetector;
	//pthread_t task;
	//pthread_create(task, 0, func.trfunc, this);
	//pthread_detach(task);
	   
	MotionDetector motionDetector;
	//std::thread task01(std::mem_fn(&MotionDetector::GmmDetector),motionDetector); 
	//std::thread task02(thread02, 5);
	std::thread thread01(bind(&MotionDetector::GmmDetector, &motionDetector));
	std::thread thread02(bind(&MotionDetector::Warn, &motionDetector));
	
	thread01.join();
	thread02.join();
	//TODO£º»¥²»Ó°Ïì
	//motionDetector.GmmDetector();
	return true;
}