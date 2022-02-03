#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
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


int hmin = 24, hmax = 0, smin = 113, smax = 130, vmin = 243, vmax = 217;

vector<int> HSVmask{ {hmin, smin, vmin, hmax, smax, vmax} };


void preProcessing() {

	cvtColor(imgOriginal, imgGray, COLOR_BGR2GRAY);
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

	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 1000)
		{
			double peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);

			if (area > maxArea && conPoly[i].size() == 4) {

				//drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 5);
				biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
				maxArea = area;
			}
			/*drawContours(imgOriginal, conPoly, i, Scalar(255, 0, 255), 2);
			rectangle(imgOriginal, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);*/
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


void checkColor(Mat img)
{
	Mat HSV;
	cvtColor(img, HSV, COLOR_BGR2HSV);
	/*Scalar lower(HSVmask[0], HSVmask[1], HSVmask[2]);
	Scalar upper(HSVmask[3], HSVmask[4], HSVmask[5]);*/

	Scalar lower(hmin, hmax, smin);
	Scalar upper(smax, vmin, vmax);
	Mat mask;
	inRange(HSV, lower, upper, mask);
	imshow("mask", mask);
	vector<Point> initialPoints = getContours(mask);
	if (isOKcolor(mask))
	{
		imshow("res", img);
	}
}




int main() {

	vector<Point> initialPoints, docPoints;
	String recPath = "nagrania/";

	VideoCapture cap(recPath + "6.mp4");
	
	HSVtrackbars();

	while (true) {
		waitKey(15);

		cap.read(imgOriginal);
		resize(imgOriginal, imgOriginal, Size(), 0.5, 0.5);


		/*imshow("Image", imgOriginal);
		waitKey(250);
		checkColor(imgOriginal);
		continue;*/


		int height = imgOriginal.size[0];
		int width = imgOriginal.size[1];

		preProcessing();
		//imshow("xd", imgErode);
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

		checkColor(imgCrop);

		imshow("Image", imgOriginal);
		//imshow("Image Dilation", imgThre);
		//imshow("Image Warp", img.Warp);
		

		//imshow("Image Crop", imgCrop);
	}
	return 0;
}