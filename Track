#include <QCoreApplication>
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

//对轮廓按面积降序排列
bool biggerSort(cv::vector<cv::Point> v1, cv::vector<cv::Point> v2)
{
	return cv::contourArea(v1)>cv::contourArea(v2);
}
int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);
	//视频不存在，就返回
	cv::VideoCapture cap("C:/Users/Xiaolong Yang/Desktop/test/x64/Debug/ten.avi");//使用视频
	//cv::VideoCapture cap(0)   //打开摄像头
	if (cap.isOpened() == false)
		return 0;

	//定义变量
	int i;

	cv::Mat frame;			//当前帧
	cv::Mat foreground;		//前景
	cv::Mat bw;				//中间二值变量
	cv::Mat se;				//形态学结构元素

							//用混合高斯模型训练背景图像
	cv::BackgroundSubtractorMOG mog;
	for (i = 0; i<100; ++i)
	{
		cout << "正在训练背景:" << i << endl;
		cap >> frame;
		if (frame.empty() == true)
		{
			cout << "视频帧太少，无法训练背景" << endl;
			getchar();
			return 0;
		}
		mog(frame, foreground, 0.01);
	}

	//目标外接框、生成结构元素（用于连接断开的小目标）
	cv::Rect rt;
	se = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	//统计目标直方图时使用到的变量
	cv::vector<cv::Mat> vecImg;
	cv::vector<int> vecChannel;
	cv::vector<int> vecHistSize;
	cv::vector<float> vecRange;
	cv::Mat mask(frame.rows, frame.cols, cv::DataType<uchar>::type);
	//变量初始化
	vecChannel.push_back(0);
	vecHistSize.push_back(32);
	vecRange.push_back(0);
	vecRange.push_back(180);

	cv::Mat hsv;		//HSV颜色空间，在色调H上跟踪目标（camshift是基于颜色直方图的算法）
	cv::MatND hist;		//直方图数组
	double maxVal;		//直方图最大值，为了便于投影图显示，需要将直方图规一化到[0 255]区间上
	cv::Mat backP;		//反射投影图
	cv::Mat result;		//跟踪结果

						//视频处理流程

	cv::vector<cv::vector<cv::Point> > contours;
	while (1)
	{
		//读视频
		cap >> frame;
		if (frame.empty() == true)
			break;

		//生成结果图
		frame.copyTo(result);

		//检测目标
		mog(frame, foreground, 0.01);
		cv::imshow("混合高斯检测前景", foreground);
	//	cv::moveWindow("混合高斯检测前景", 400, 0);
		//对前景进行中值滤波、形态学膨胀操作，以去除伪目标和接连断开的小目标（一个大车辆有时会断开成几个小目标）
		cv::medianBlur(foreground, foreground, 5);
		cv::imshow("中值滤波", foreground);
	//	cv::moveWindow("中值滤波", 800, 0);
		cv::morphologyEx(foreground, foreground, cv::MORPH_DILATE, se);

		//检索前景中各个连通分量的轮廓
		foreground.copyTo(bw);
		//cv::vector<cv::vector<cv::Point> > contours;
		//contours.clear();
		cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		if (contours.size()<1)
			continue;
		//对连通分量进行排序
		std::sort(contours.begin(), contours.end(), biggerSort);

		//结合camshift更新跟踪位置（由于camshift算法在单一背景下，跟踪效果非常好；
		//但是在监控视频中，由于分辨率太低、视频质量太差、目标太大、目标颜色不够显著
		//等各种因素，导致跟踪效果非常差。  因此，需要边跟踪、边检测，如果跟踪不够好，
		//就用检测位置修改
		cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
		vecImg.clear();
		vecImg.push_back(hsv);
		for (int k = 0; k<contours.size(); ++k)
		{
			//第k个连通分量的外接矩形框
			if (cv::contourArea(contours[k])<cv::contourArea(contours[0]) / 5)
				break;
			rt = cv::boundingRect(contours[k]);
			//mask=0;
			mask(rt) = 255;

			//统计直方图
			cv::calcHist(vecImg, vecChannel, mask, hist, vecHistSize, vecRange);
			cv::minMaxLoc(hist, 0, &maxVal);
			hist = hist * 255 / maxVal;
			//计算反向投影图
			cv::calcBackProject(vecImg, vecChannel, hist, backP, vecRange, 1);
			//camshift跟踪位置
			cv::Rect search = rt;
			cv::RotatedRect rrt = cv::CamShift(backP, search, cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 1));
			cv::Rect rt2 = rrt.boundingRect();
			rt &= rt2;

			//跟踪框画到视频上
			cv::rectangle(result, rt, cv::Scalar(0, 255, 0), 2);
		}

		//结果显示
		cv::imshow("Origin", frame);
	//	cv::moveWindow("Origin", 0, 0);

		cv::imshow("膨胀运算", foreground);
	//	cv::moveWindow("膨胀运算", 0, 300);

		cv::imshow("反向投影", backP);
	//	cv::moveWindow("反向投影", 400, 300);

		cv::imshow("跟踪效果", result);
	//	cv::moveWindow("跟踪效果", 800, 300);
		cv::waitKey(30);
	}

	getchar();
	//  return 0;

	return a.exec();
}
