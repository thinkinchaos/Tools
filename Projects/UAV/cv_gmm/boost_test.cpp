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
//	// ����һ���̳߳أ���ʼ��Ϊ2���߳�  
//	pool tp(2);
//
//	// ����4���̺߳���  
//	tp.schedule(&first_task); // ����������ִ�к���  
//	tp.wait(); <span style = "white-space:pre">	< / span>//�ȴ�ǰ����̺߳���ִ�н����ż���ִ�к�����߳�
//		tp.schedule(&second_task);
//	tp.wait();
//	tp.schedule(&third_task);
//	tp.schedule(boost::bind(task_with_parameter, 8, "hello")); // ������������ִ�к���  
//	tp.schedule(&third_task);
//	// ���������ȵ�4���߳�ȫ��ִ�н����Ż��˳�  
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