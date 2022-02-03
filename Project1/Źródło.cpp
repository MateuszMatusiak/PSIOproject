#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat imgOriginal;
Mat imgGray;
Mat imgBlur;
Mat imgCanny;
Mat imgThre;
Mat imgDilate;
Mat imgErode;
Mat imgWarp;
Mat imgCrop;

VideoCapture cap;
vector<Point> MaxContour;
bool isContour = false;
vector<vector<Point>> MaxConPoly;
vector<Rect> MaxBoundRect;

Mat Result;
int clipIndex = 0;

//int hmin = 24, hmax = 0, smin = 113, smax = 130, vmin = 243, vmax = 217;
// 3 int hmin = 49, hmax = 0, smin = 55, smax = 99, vmin = 255, vmax = 99;
 int hmin = 31, hmax = 6, smin = 119, smax = 102, vmin = 242, vmax = 199;

vector<vector<int>> HSVmask{	{hmin, smin, vmin, hmax, smax, vmax},
								{hmin, smin, vmin, hmax, smax, vmax}, 
								{hmin, smin, vmin, hmax, smax, vmax}, 
								{hmin, smin, vmin, hmax, smax, vmax}, 
								{hmin, smin, vmin, hmax, smax, vmax}, 
								{hmin, smin, vmin, hmax, smax, vmax}, };

void preProcessing(Mat x) {

	cvtColor(x, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	Canny(imgBlur, imgCanny, 25, 75);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDilate, kernel);
	erode(imgDilate, imgErode, kernel);
}

vector<Point> getContours(Mat x) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(x, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(x, contours, -1, Scalar(255, 0, 255), 2);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	vector<Point> biggest;
	double maxArea = 0;

	int maxIndex = -1;
	
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
			maxArea = area;
			maxIndex = i;
		}
	}

	int i = maxIndex;
	if (maxArea > 1000)
	{
		cout << maxArea << endl;
		MaxContour = contours[i];
		isContour = true;
		MaxConPoly = vector<vector<Point>>(contours.size());
		MaxBoundRect = vector<Rect>(contours.size());
	}

	if (isContour) {
		double peri = arcLength(MaxContour, true);
		approxPolyDP(MaxContour, MaxConPoly[i], 0.02 * peri, true);

		if (MaxConPoly[i].size() == 4) {

			//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
			drawContours(imgOriginal, MaxConPoly, i, Scalar(255, 0, 255), 2);
			biggest = { MaxConPoly[i][0],MaxConPoly[i][1] ,MaxConPoly[i][2] ,MaxConPoly[i][3] };
			rectangle(imgOriginal, MaxBoundRect[i].tl(), MaxBoundRect[i].br(), Scalar(0, 255, 0), 5);
		}
	}

	return biggest;
}

bool isOKcolor(Mat x) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(x, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(x, contours, -1, Scalar(255, 0, 255), 2);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());


	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		cout << area<< endl;
		if (area > 370000&&area<500000)
		{
				return true;
		}
	}
	return false;
}

vector<Point> reorder(vector<Point> points)
{
	vector<Point> newPoints;
	vector<int>  sumPoints, subPoints;

	for (int i = 0; i < 4; i++)
	{
		sumPoints.push_back(points[i].x + points[i].y);
		subPoints.push_back(points[i].x - points[i].y);
	}

	newPoints.push_back(points[min_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //0
	newPoints.push_back(points[max_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //1
	newPoints.push_back(points[min_element(subPoints.begin(), subPoints.end()) - subPoints.begin()]); //2
	newPoints.push_back(points[max_element(sumPoints.begin(), sumPoints.end()) - sumPoints.begin()]); //3

	return newPoints;
}

void getWarp(vector<Point> points, float width, float height)
{
	Point2f src[4] = { points[0],
					   points[1],
					   points[2],
					   points[3] };

	Point2f dst[4] = { {0.0f,	0.0f},
					   {width,	0.0f},
					   {0.0f,	height},
					   {width,	height} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(imgOriginal, imgWarp, matrix, Point(width, height));
	return;
}

void HSVtrackbars() {

	namedWindow("Trackbars", (640, 200));
	createTrackbar("Hue Min", "Trackbars", &hmin, 179);
	createTrackbar("Hue Max", "Trackbars", &hmax, 179);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);
}


Mat checkColor(Mat img)
{
	Mat HSV;
	cvtColor(img, HSV, COLOR_BGR2HSV);
	/*Scalar lower(HSVmask[0], HSVmask[1], HSVmask[2]);
	Scalar upper(HSVmask[3], HSVmask[4], HSVmask[5]);*/

	Scalar lower(hmin, hmax, smin);
	Scalar upper(smax, vmin, vmax);
	Mat mask;
	inRange(HSV, lower, upper, mask);
	//imshow("mask", mask);
	//vector<Point> initialPoints = getContours(mask);
	/*if (isOKcolor(mask))
	{
		imshow("res", img);
	}*/

	return mask;
}



Mat getClip(int index = -1) {
	Mat toReturn;

	if (index >= 0) {
		++clipIndex;
		if (clipIndex > index) {
			Result.copyTo(toReturn);
			return toReturn;
		}
	}
	cap.read(Result);
	resize(Result, Result, Size(), 0.5, 0.5);
	Result.copyTo(toReturn);
	return toReturn;
}

void LoadVideo(string name) {
	String recPath = "nagrania/";
	cap = VideoCapture(recPath + name);
}

int main() {

	vector<Point> initialPoints, docPoints;
	
	LoadVideo("4.mp4");
	
	HSVtrackbars();
	int i = 0;
	while (true) {
		++i;
		imgOriginal = getClip();

		waitKey(1);


		/*
			/*ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
			ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
			ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
			ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)*/

			

		//Nie tykaæ
		//-------------------------------------
		//imshow("Image", imgOriginal);
		////waitKey(10);
		//checkColor(imgOriginal);
		//continue;
		//-------------------------------------

		int height = imgOriginal.size[0];
		int width = imgOriginal.size[1];

		/*Mat hsv;
		cvtColor(imgOriginal, hsv, COLOR_BGR2HSV);

		for (int j = 0; j < imgOriginal.rows; j++)
			for (int i = 0; i < imgOriginal.cols; i++)
				hsv.at<Vec3b>(j, i)[1] = 255;

		cvtColor(hsv, imgOriginal, COLOR_HSV2BGR);*/

		//dilate(imgOriginal, imgOriginal, Mat::ones(3, 3, 0)); //Mat.ones(new Size(3, 3), 0)

		Mat grayC;
		cvtColor(imgOriginal, grayC, COLOR_RGB2GRAY);
		Mat imgToProc;
		imgOriginal.copyTo(imgToProc);
		//threshold(grayC, imgToProc, 180, 180, THRESH_TRUNC);


		for (int j = 0; j < grayC.rows; j++)
			for (int i = 0; i < grayC.cols; i++)
			{
				int x = grayC.at<uchar>(j, i);
				if (x > 180) {
					grayC.at<uchar>(j, i) = 50;
					imgToProc.at<Vec3b>(j, i)[0] = 50;
					imgToProc.at<Vec3b>(j, i)[1] = 50;
					imgToProc.at<Vec3b>(j, i)[2] = 50;
				}
				if (x < 50) {
					grayC.at<uchar>(j, i) = 50;
					imgToProc.at<Vec3b>(j, i)[0] = 50;
					imgToProc.at<Vec3b>(j, i)[1] = 50;
					imgToProc.at<Vec3b>(j, i)[2] = 50;
				}
			}

		//imshow("proc", imgToProc);
		
		preProcessing(imgOriginal);
		imshow("Erode1", imgErode);
		
		
		Mat mask = checkColor(imgOriginal);
		Mat maskCopy;
		mask.copyTo(maskCopy);

		int sigma = 2;
		for (int j = 0; j < mask.rows; j++)
			for (int i = 0; i < mask.cols; i++)
			{
				if (mask.at<bool>(j, i) != 0)
				{
					for (int k = -sigma; k < sigma; ++k)
						for (int l = -sigma; l < sigma; ++l)
						{
							int y = j + k;
							int x = i + l;
							if (x < 0 || y < 0 || x > mask.cols - 1 || y > mask.rows - 1)
								continue;
							maskCopy.at<uchar>(y, x) = 255;
						}
				}
			}

		//Mat kernel2 = getStructuringElement(MORPH_RECT, Size(8, 8));
		//dilate(mask, mask, kernel2);




		imshow("Mask", maskCopy);

		for (int j = 0; j < grayC.rows; j++)
			for (int i = 0; i < grayC.cols; i++)
			{
				if (maskCopy.at<bool>(j, i) == 0)
					imgErode.at<bool>(j, i) = 0;
			}

		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(imgErode, imgErode, kernel);

		imshow("Erode2", imgErode);

		initialPoints = getContours(imgErode);
		if (initialPoints.size() < 4) {
			imshow("Image", imgOriginal);
			continue;
		}
		docPoints = reorder(initialPoints);
		getWarp(docPoints, (float)width, (float)height);

		int cropVal = 5;
		Rect roi(cropVal, cropVal, width - (2 * cropVal), height - (2 * cropVal));
		imgCrop = imgWarp(roi);

		//checkColor(imgCrop);

		imshow("Image", imgOriginal);
		//imshow("Image Dilation", imgThre);
		//imshow("Image Warp", img.Warp);
		

		imshow("Image Crop", imgCrop);
	}
	return 0;
}
