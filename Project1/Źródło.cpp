#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#define PATH "2.mp4"

using namespace cv;
using namespace std;

class MaxArea {
public:
	Mat mat;

	bool isContour;
	int idx;
	double area;

	double peri;

	vector<Point> contour;
	vector<Point> conPoly;
	vector<Point> wrong;
	Rect rect;

	MaxArea() {
		this->isContour = false;
		this->idx = 0;
		this->area = 0;
	}

	MaxArea(vector<vector<Point>> contours, Mat mat) {
		this->mat = mat;
		this->isContour = false;
		this->idx = 0;
		this->area = 0;
		Set(contours);
	}

	MaxArea(const MaxArea& maxArea) {
		mat = maxArea.mat;
		isContour = maxArea.isContour;
		idx = maxArea.idx;
		area = maxArea.area;
		contour = maxArea.contour;
		conPoly = maxArea.conPoly;
		rect = maxArea.rect;
	}

	void Reset() {
		isContour = false;
		idx = 0;
		area = 0;

		peri = 0;

		contour = vector<Point>();
		conPoly = vector<Point>();;
		wrong = vector<Point>();;
		rect = Rect();
	}

	void Set(vector<vector<Point>> contours) {
		int size = contours.size();

		if (size <= 0)
			return;

		idx = 0;
		for (int i = 0; i < size; i++)
		{
			double tempArea = contourArea(contours[i]);
			if (tempArea > area)
			{
				area = tempArea;
				idx = i;
			}
		}

		this->contour = contours[idx];
	}

	void UpdateMax(int size) {
		this->conPoly = vector<Point>(size);
		this->rect = Rect();
		this->isContour = true;
		peri = arcLength(contour, true);
	}

	vector<Point> getConPolyVector() {
		if (!isContour)
			return vector<Point>();

		approxPolyDP(contour, conPoly, 0.02 * peri, true);

		wrong = conPoly;
		conPoly = Optimize(conPoly);

		return conPoly;
	}


	struct line {
		Point A;
		Point B;
		bool out;

		line(Point A, Point B) {
			this->A = A;
			this->B = B;
			out = false;
		}

		double length()
		{
			return sqrt(pow((B.x - A.x), 2) + pow((B.y - A.y), 2));
		}
	};

	bool ccw(Point A, Point  B, Point  C) {
		return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
	}

	bool intersect(line l1, line l2) {
		Point A = l1.A;
		Point B = l1.B;
		Point C = l2.A;
		Point D = l2.B;
		return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
	}

	vector<Point> findMax(Mat x) {
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;

		findContours(x, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
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
					biggest = { conPoly[i][0],conPoly[i][1] ,conPoly[i][2] ,conPoly[i][3] };
					maxArea = area;
				}
			}
		}
		return biggest;
	}

	vector<Point> Optimize(vector<Point> p) {
		vector<line> l;
		for (int i = 0; i < p.size() - 1; ++i)
			for (int j = i + 1; j < p.size(); ++j) {
				l.push_back(line(p[i], p[j]));
			}

		Mat res(mat.size(), CV_8UC3, cv::Scalar(0, 0, 0));

		for (int i = 0; i < l.size(); ++i)
			for (int j = i + 1; j < l.size(); ++j) {
				cv::line(res, l[i].A, l[i].B, Scalar(0, 255, 255), 1);
				cv::line(res, l[j].A, l[j].B, Scalar(0, 255, 255), 1);

				if (intersect(l[i], l[j]))
				{
					if (l[i].A != l[j].A
						&& l[i].A != l[j].B
						&& l[i].B != l[j].B
						&& l[i].B != l[j].A) {
						l[i].out = true;
						l[j].out = true;
					}
				}
			}

		cvtColor(res, res, COLOR_BGR2GRAY);

		return findMax(res);
	}
};

int currentVideo = 0;
int prevVideo = 0;

VideoCapture cap;

Mat Result;
int clipIndex = 0;

Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
MaxArea maxArea = MaxArea();

int b = 1;
int sigma = 6;
int areaLimit = 10000;

int reflectionMin = 50;
int reflectionMax = 180;

int hmin = 19, hmax = 6, smin = 121, smax = 99, vmin = 213, vmax = 238;

Mat preProcessing(Mat source) {
	Mat gray;
	Mat blur;
	Mat canny;
	Mat Dilate;
	Mat Erode;

	cvtColor(source, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(3, 3), 3, 0);
	Canny(blur, canny, 25, 75);
	dilate(canny, Dilate, kernel);
	erode(Dilate, Erode, kernel);
	return Erode;
}

Mat getWarp(Mat source, vector<Point> points, float width, float height)
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
	Mat imgWarp;
	warpPerspective(source, imgWarp, matrix, Point(width, height));
	return imgWarp;
}

void HSVtrackbars() {

	namedWindow("Trackbars", (640, 200));
	createTrackbar("Hue Min", "Trackbars", &hmin, 255);
	createTrackbar("Hue Max", "Trackbars", &hmax, 255);
	createTrackbar("Sat Min", "Trackbars", &smin, 255);
	createTrackbar("Sat Max", "Trackbars", &smax, 255);
	createTrackbar("Val Min", "Trackbars", &vmin, 255);
	createTrackbar("Val Max", "Trackbars", &vmax, 255);
	createTrackbar("ST.....", "Trackbars", &b, 1);
	createTrackbar("Sigma..", "Trackbars", &sigma, 6);
	createTrackbar("ReflectionMin", "Trackbars", &reflectionMin, 255);
	createTrackbar("ReflectionMax", "Trackbars", &reflectionMax, 255);
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
	if (b) {
		cap.read(Result);
		resize(Result, Result, Size(), 0.5, 0.5);
	}
	Result.copyTo(toReturn);
	return toReturn;
}

void LoadVideo(string name) {
	String recPath = "nagrania/";
	cap = VideoCapture(recPath + name);
}

Mat getMask(Mat img)
{
	Mat HSV;
	cvtColor(img, HSV, COLOR_BGR2HSV);

	Scalar lower(hmin, hmax, smin);
	Scalar upper(smax, vmin, vmax);
	Mat mask;
	inRange(HSV, lower, upper, mask);
	return mask;
}

vector<Point> getContours(Mat source) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	Mat temp;
	Result.copyTo(temp);

	findContours(source, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	MaxArea area(contours, source);

	if (area.area > areaLimit) {
		maxArea = area;
		maxArea.UpdateMax(contours.size());
	}

	return maxArea.getConPolyVector();
}

void RemoveReflections(Mat source) {
	Mat gray;
	cvtColor(source, gray, COLOR_RGB2GRAY);

	for (int j = 0; j < gray.rows; j++)
		for (int i = 0; i < gray.cols; i++)
		{
			int x = gray.at<uchar>(j, i);
			if (x > reflectionMax) {
				gray.at<uchar>(j, i) = reflectionMax;
				source.at<Vec3b>(j, i)[0] = reflectionMax;
				source.at<Vec3b>(j, i)[1] = reflectionMax;
				source.at<Vec3b>(j, i)[2] = reflectionMax;
			}
			if (x < reflectionMin) {
				gray.at<uchar>(j, i) = reflectionMin;
				source.at<Vec3b>(j, i)[0] = reflectionMin;
				source.at<Vec3b>(j, i)[1] = reflectionMin;
				source.at<Vec3b>(j, i)[2] = reflectionMin;
			}
		}
}

Mat Marge(Mat mask, Mat Erode) {
	Mat result;
	Erode.copyTo(result);
	for (int j = 0; j < mask.rows; j++)
		for (int i = 0; i < mask.cols; i++)
			if (mask.at<uchar>(j, i) < 128)
				result.at<uchar>(j, i) = 0;
	return result;
}

bool isOKcolor(Mat x) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(x, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());


	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		if (area > 370000 && area < 500000)
		{
			return true;
		}
	}
	return false;
}

int main() {

	vector<Point> initialPoints, docPoints;
	Mat Original;

	namedWindow("Trackbars", 0x00000100);
	createTrackbar("Video", "Trackbars", &currentVideo, 2);

	//HSVtrackbars();

	LoadVideo(PATH);

	while (true) {

		if (currentVideo != prevVideo) {
			destroyAllWindows();
			namedWindow("Trackbars", 0x00000100);
			createTrackbar("Video", "Trackbars", &currentVideo, 2);
			LoadVideo(to_string(currentVideo + 1) + ".mp4");
			prevVideo = currentVideo;
		}

		Original = getClip();
		waitKey(30);

		Mat withoutReflections;
		Original.copyTo(withoutReflections);

		RemoveReflections(withoutReflections);

		Mat Erode = preProcessing(withoutReflections);

		Mat mask = getMask(Original);
		Mat dilateMask;
		mask.copyTo(dilateMask);

		Mat kernel2 = getStructuringElement(MORPH_RECT, Size(sigma + 1, sigma + 1));
		dilate(mask, dilateMask, kernel2);

		Mat marged = Marge(dilateMask, Erode);

		initialPoints = getContours(marged);
		imshow("Image", Original);

		if (initialPoints.size() != 4) {
			continue;
		}

		convexHull(initialPoints, docPoints);
		auto p = docPoints;
		docPoints = { p[0],p[1],p[3],p[2] };

		Mat imgWarp = getWarp(Original, docPoints, (float)Original.cols, (float)Original.rows);

		int cropVal = 5;
		Rect roi(cropVal, cropVal, Original.cols - (2 * cropVal), Original.rows - (2 * cropVal));

		Mat imgCrop = imgWarp(roi);

		if (isOKcolor(getMask(imgCrop))) {
			imshow("Result", imgCrop);
			maxArea.Reset();
		}
	}
	return 0;
}
