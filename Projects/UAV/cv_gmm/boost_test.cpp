//#include <iostream>  
//#include <boost/bind.hpp>  
//#include <boost/threadpool.hpp>  
//using namespace std;
//using namespace boost::threadpool;
//
//void first_task()
//{
//	for (int i = 0; i < 30; ++i)
//		cout << "first" << i << endl;
//}
//void second_task()
//{
//	for (int i = 0; i < 30; ++i)
//		cout << "second" << i << endl;
//}
//void third_task()
//{
//	for (int i = 0; i < 30; ++i)
//		cout << "third" << i << endl;
//}
//void task_with_parameter(int value, string str)
//{
//	printf("task_with_parameter with int=(%d).\n", value);
//	printf("task_with_parameter with string=(%s).\n", str.c_str());
//}
//void execute_with_threadpool()
//{
//	// 创建一个线程池，初始化为2个线程  
//	pool tp(2);
//
//	// 调用4个线程函数  
//	tp.schedule(&first_task); // 不带参数的执行函数  
//	tp.wait(); <span style = "white-space:pre">	< / span>//等待前面的线程函数执行结束才继续执行后面的线程
//		tp.schedule(&second_task);
//	tp.wait();
//	tp.schedule(&third_task);
//	tp.schedule(boost::bind(task_with_parameter, 8, "hello")); // 带两个参数的执行函数  
//	tp.schedule(&third_task);
//	// 这儿函数会等到4个线程全部执行结束才会退出  
//	while (1)
//	{
//		;
//	}
//}
//
//int main()
//{
//	execute_with_threadpool();
//	return 0;
//}
