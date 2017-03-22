#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>
#include <omp.h>
#include <iostream>

#define euler 2.71828182845904523536
#define pi 3.14159265359
#define pi2 6.28318531
#define gaussFilterSize 5
#define gaussSigma 3
#define gaussWeightSSDSigma 3

#define cornerResponseThreshold 0.03 // less -> more points
#define SSDThreshold 0.3// lower -> stricter matching
#define RatioThreshold 0.4 // lower -> more distinction between best /second best



// TRY TO ADD ROTATIONAL INVARIANCE ////////////////////
struct Feature {
	cv::Point point;
	std::vector<float> descriptor;
};
struct PointPairDistance {
	cv::Point point1;
	cv::Point point2;
	float SSDdistance;
};
struct PointPairRatio {
	cv::Point point1;
	cv::Point bestMatch;
	cv::Point secondBestMatch;
	float ratio;

};
// returns of a value for the specific gaussian cell in the 
float gaussian(int x, int y, double sigma) {
	double s2 = sigma*sigma;
	double exp = -((x*x) + (y*y)) / (2 * s2);
	return (float)(1 / (2 * CV_PI*s2))*cv::pow(euler, exp);
}
// we pass a reference to a mat and turn that mat into a gaussian kernel
void GaussianKernel(int sizex, int sizey, double sigma, cv::Mat &kernel) {
	int xmin = -(sizex) / 2;
	int xmax = -xmin;
	int xcenter = xmax;

	int ymin = -sizey / 2;
	int ymax = -ymin;
	int ycenter = ymax;

	cv::Mat kern;
	kern.create(sizex, sizey, CV_32FC1);
	float sum = 0;

	for (int x = xmin; x <= xmax; x++) {
		for (int y = ymin; y <= ymax;y++) {
			kern.at<float>(x + xcenter, y + ycenter) = gaussian(x, y, sigma);
			sum += kern.at<float>(x + xcenter, y + ycenter);
		}
	}
	kern = kern / sum;
	kernel = kern;
}

// we get the value of the cell, if cell is OOB, we return 0
float getCellValue(cv::Mat &input, int x, int y) {
	if (x >= input.rows || y >= input.cols || x < 0 || y < 0)
		return 0;
	else {
		return input.at<float>(x, y);
	}
}

// we use filter2d to apply a gaussian filter on an Mat
void applyGaussianFilter(cv::Mat &input, cv::Mat &output, int sizeX, int sizeY, double sigma) {

	cv::Mat guassFilter;
	GaussianKernel(sizeX, sizeY, sigma, guassFilter);
	cv::filter2D(input, output, -1, guassFilter);

}

// apply sobel operator on image in x direction
void applySobelX(cv::Mat &input, cv::Mat &output) {
	cv::Mat sobelOp = (cv::Mat_<float>(3, 3) <<
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1);

	cv::filter2D(input, output, -1, sobelOp);
}

// apply sobel operator on image in y direction
void applySobelY(cv::Mat &input, cv::Mat &output) {
	cv::Mat sobelOp = (cv::Mat_<float>(3, 3) <<
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1);

	cv::filter2D(input, output, -1, sobelOp);
}

// we generate a harris matrix for a given pixle using 3 inputs, IxIx, IyIy, IxIy
cv::Mat makeHarrisMatrix(cv::Mat &Ixx, cv::Mat &Iyy, cv::Mat &Ixy, int x, int y) {
	cv::Mat harrisMatrix = (cv::Mat_<float>(2, 2) <<
		getCellValue(Ixx, x, y), getCellValue(Ixy, x, y),
		getCellValue(Ixy, x, y), getCellValue(Iyy, x, y));
	return harrisMatrix;
}

// we calculate the corner response value for a given pixle
float getCornerResponseValue(cv::Mat &Ixx, cv::Mat &Iyy, cv::Mat &Ixy, int x, int y)
{
	cv::Mat harrisMatrix = makeHarrisMatrix(Ixx, Iyy, Ixy, x, y);
	float det = harrisMatrix.at<float>(0, 0) * harrisMatrix.at<float>(1, 1) - harrisMatrix.at<float>(0, 1) * harrisMatrix.at<float>(1, 0);
	float trace = harrisMatrix.at<float>(0, 0) + harrisMatrix.at<float>(1, 1);
	return det / trace;

}

// we create a Mat of all corner response values and returns a list of all points that are greater than threshold value
std::vector<cv::Point> setResponseMat(cv::Mat &Ixx, cv::Mat &Iyy, cv::Mat &Ixy, cv::Mat &cornerResponseMat) {
	std::vector<cv::Point> list(0);
	cornerResponseMat.create(Ixx.rows, Ixx.cols, CV_32FC1);
	for (int i = 0; i < Ixx.rows;i++) {
		for (int j = 0; j < Ixx.cols;j++) {
			double c = getCornerResponseValue(Ixx, Iyy, Ixy, i, j);
			if (c > cornerResponseThreshold) {
				cornerResponseMat.at<float>(i, j) = c;
				list.push_back(cv::Point(i, j));
			}
			else
				cornerResponseMat.at<float>(i, j) = 0;
		}
	}
	return list;
}

// return true if pixle is local max within a window
bool isLocalMax(cv::Point point, cv::Mat &inputMat, int xdim, int ydim) {
	float center = inputMat.at<float>(point.x, point.y);
	int xmin = -xdim / 2;
	int xmax = -xmin;

	int ymin = -ydim / 2;
	int ymax = -ymin;

	for (int i = xmin; i <= xmax;i++) {

		for (int j = ymin; j <= ymax;j++) {

			if (getCellValue(inputMat, point.x + i, point.y + j) > center) {
				return false;
			}
		}
	}
	return true;
}

// filters image to show only local maxima and returns vector with coresponding points
std::vector<cv::Point> filterLocalMaxOnly(cv::Mat &inputMat, cv::Mat &outputMat, std::vector<cv::Point>& pointList, int windowSizeX, int windowSizeY) {
	outputMat.create(inputMat.rows, inputMat.cols, CV_32FC1);
	std::vector<cv::Point> localMaxima(0);
#pragma parallel for
	for (int i = 0; i < pointList.size();i++) {

		if (isLocalMax(pointList[i], inputMat, windowSizeX, windowSizeY)) {
			localMaxima.push_back(pointList[i]);
			outputMat.at<float>(pointList[i].x, pointList[i].y) = 0.5 + inputMat.at<float>(pointList[i].x, pointList[i].y);
		}
		else {

			outputMat.at<float>(pointList[i].x, pointList[i].y) = 0;
		}
	}
	return localMaxima;
}

std::vector<int> getRectDimensions(cv::Mat &image, cv::Point point, int width, int height) {
	std::vector<int> dims(4);

	int xmin = -width / 2;
	int xmax = -xmin;
	xmin += point.x;
	xmax += point.x;

	int ymin = -height / 2;
	int ymax = -ymin;

	ymin += point.y;
	ymax += point.y;



	//xstart
	if (xmin< 0)
		dims[0] = 0;
	else
		dims[0] = xmin;
	//ystart
	if (ymin < 0)
		dims[1] = 0;
	else
		dims[1] = ymin;

	//xend
	if (xmax > image.rows)
		dims[2] = image.rows - point.x;
	else
		dims[2] = width;

	//yend
	if (ymax > image.cols)
		dims[3] = image.cols - point.y;
	else
		dims[3] = height;

	return dims;
}


std::vector<float> getSiftDescriptorHistogram(cv::Mat dx, cv::Mat dy, cv::Point point) {
	cv::Mat gaussian17x17;
	GaussianKernel(17, 17, gaussWeightSSDSigma, gaussian17x17);
	//histogram vector  (will have size of 128 by end)
	std::vector<float> histogram(128);
	int step = 0;
	//to check angle ranges
	float angleDelta = 2 * pi / 8.0f;
	float rotationOffset = atan(getCellValue(dy, point.x, point.y) / getCellValue(dx, point.x, point.y));
	float xOffset = 0, yOffset = 0;
	//create a 16 x 16 area around point (if possible)
	std::vector<int> corners = getRectDimensions(dx, point, 16, 16);
	cv::Mat window16x16dx = dx(cv::Rect(corners[1], corners[0], corners[3], corners[2]));
	cv::Mat window16x16dy = dy(cv::Rect(corners[1], corners[0], corners[3], corners[2]));

	for (int i = 0; i < window16x16dx.rows - 3;i += 4) {
		for (int j = 0; j < window16x16dx.cols - 3; j += 4) {
			// store 4x4 histogram in this vector
			//std::vector<float> tempHistogram(8);

			if (i + 2 <= window16x16dx.rows && j + 2 <= window16x16dx.cols) {

				// create 4x4 sub matrices inside 16x16 region (if possible)
				std::vector<int> innerCorners = getRectDimensions(window16x16dx, cv::Point(i + 2, j + 2), 4, 4);
				cv::Mat window4x4dx = window16x16dx(cv::Rect(innerCorners[1], innerCorners[0], innerCorners[3], innerCorners[2]));
				cv::Mat window4x4dy = window16x16dy(cv::Rect(innerCorners[1], innerCorners[0], innerCorners[3], innerCorners[2]));

				// compute direction value of each element in cell and store in vector
#pragma omp parallel for
				for (int x = 0; x < window4x4dx.rows;x++) {
					for (int y = 0; y < window4x4dx.cols;y++) {
						// use atan of dy / dx to find angle of direction
						float dx = getCellValue(window4x4dx, x, y);
						float dy = getCellValue(window4x4dy, x, y);
						float theta = atan(dy / dx) - rotationOffset;
						int multiplierX = i + x, multiplierY = j + x;

						if (theta < 0)
							theta = pi2 + theta;
						for (int delta = 0; delta < 8; delta++) {
							if (theta >= delta * angleDelta && theta < (delta + 1) *angleDelta) {

								histogram[delta + step] += sqrt(dy*dy + dx*dx) * getCellValue(gaussian17x17, multiplierX, multiplierY);
								break;
							}
						}
					}
				}
			}
			//concatinate 8 dimensional vector to final vector list
			//histogram.insert(histogram.end(), tempHistogram.begin(), tempHistogram.end());
			step += 8;
		}
	}
	// normalizing vector to have magnitute 1, and cap max value at 0.20
	float squareSum = 0;
	for (int i = 0; i < histogram.size();i++) {

		squareSum += histogram[i] * histogram[i];
	}
	float rootSum = sqrt(squareSum);
	float normalizeFactor = 1 / rootSum;

	for (int i = 0; i < histogram.size();i++) {
		histogram[i] *= normalizeFactor;
		if (histogram[i] > 0.2) {
			histogram[i] = 0.2;
		}
	}

	return histogram;
}
// superimpose interest points over origional image for testing
cv::Mat highlightPoints(cv::Mat &input, std::vector<cv::Point> points, int radius) {
	cv::Mat output;
	input.copyTo(output);
	for (int i = 0; i < points.size();i++) {
		const cv::Scalar color(0, 0, 255, 0.5);
		cv::Point point(points[i].y, points[i].x);
		cv::circle(output, point, radius, color, 1, 8, 0);
	}
	return output;
}

std::vector<Feature> getFeatureVector(std::string imagePath, cv::Mat &output) {
	using namespace cv;
	std::cout << "\nGetting feature vector for image: " << imagePath << std::endl;

	Mat colorImage = imread(imagePath, CV_LOAD_IMAGE_COLOR);
	Mat image = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);
	image.convertTo(image, CV_32FC1);
	image /= 255.0f;
	colorImage.convertTo(colorImage, CV_32FC3);
	colorImage /= 255.0f;

	Mat dx(image.rows, image.cols, image.type());
	Mat dy(image.rows, image.cols, image.type());
	Mat dxdy(image.rows, image.cols, image.type());
	Mat dxdx(image.rows, image.cols, image.type());
	Mat dydy(image.rows, image.cols, image.type());

	applySobelX(image, dx);
	applySobelY(image, dy);
	dxdy = dx.mul(dy);
	dxdx = dx.mul(dx);
	dydy = dy.mul(dy);

	applyGaussianFilter(dxdy, dxdy, gaussFilterSize, gaussFilterSize, gaussSigma);
	applyGaussianFilter(dxdx, dxdx, gaussFilterSize, gaussFilterSize, gaussSigma);
	applyGaussianFilter(dydy, dydy, gaussFilterSize, gaussFilterSize, gaussSigma);


	Mat cornerResponseMat, cornerLocalMaxOnly;
	std::vector<Point> points = setResponseMat(dxdx, dydy, dxdy, cornerResponseMat);
	points = filterLocalMaxOnly(cornerResponseMat, cornerLocalMaxOnly, points, 3, 3);
	Mat highlights = highlightPoints(colorImage, points, 1);

	std::vector<Feature> features(points.size());
#pragma omp parallel for
	for (int i = 0; i < points.size();i++) {
		Feature tempFeature;
		tempFeature.point = points[i];
		tempFeature.descriptor = getSiftDescriptorHistogram(dx, dy, points[i]);

		features[i] = tempFeature;
	}



	//output = highlights;
	output = colorImage;
	return features;
}

float getSSDValue(Feature f1, Feature f2) {
	float sum = 0;
#pragma omp parallel for 
	for (int i = 0; i < f1.descriptor.size();i++) {

		float diff = f1.descriptor[i] - f2.descriptor[i];
		diff *= diff;
		sum += diff;

	}
	return sum;
}


std::vector<PointPairDistance> getBestFeatureMatchesSSD(std::vector<Feature> &img1_features, std::vector<Feature> &img2_features) {
	std::vector<PointPairDistance> pairs(0);
	int totalSize = img1_features.size();
	int count = 0;
#pragma omp parallel for 
	for (int i = 0; i < img1_features.size();i++) {
		std::cout << "\n" << count++ << "/" << totalSize;
		float distance = SSDThreshold;
		int match = -1;
#pragma omp parallel for 
		for (int j = 0; j < img2_features.size();j++) {
			float tempDistance = getSSDValue(img1_features[i], img2_features[j]);
			if (tempDistance < distance) {
				distance = tempDistance;
				match = j;
			}
		}
		bool bestMatchInImage = true;
		if (match != -1) {
			//	#pragma omp parallel for 
			//	for (int x = i; x < img1_features.size();x++) {
			//		if (getSSDValue(img1_features[x], img2_features[match]) < distance) {
			//			bestMatchInImage = false;
			//			break;
			//		}
			//}
			if (bestMatchInImage) {
				PointPairDistance ppssd;
				ppssd.point1 = img1_features[i].point;
				ppssd.point2 = img2_features[match].point;
				ppssd.SSDdistance = distance;
				pairs.push_back(ppssd);
			}
		}
	}
	return pairs;
}
std::vector<PointPairRatio> getBestFeatureMatchesRatio(std::vector<Feature> &img1_features, std::vector<Feature> &img2_features) {
	std::vector<PointPairRatio> pairs(0);
	int totalSize = img1_features.size();
	int count = 0;
#pragma omp parallel for 
	for (int i = 0; i < img1_features.size();i++) {
		std::cout << "\n" << count++ << "/" << totalSize;

		float distanceBest = SSDThreshold;
		float distanceSecondBest = SSDThreshold;
		int bestMatch = -1;
		int secondBestMatch = -1;

		for (int j = 0; j < img2_features.size();j++) {
			float tempDistance = getSSDValue(img1_features[i], img2_features[j]);
			if (tempDistance < distanceBest || bestMatch == -1) {
				distanceSecondBest = distanceBest;
				distanceBest = tempDistance;
				secondBestMatch = bestMatch;
				bestMatch = j;
			}
			else
				if (tempDistance < distanceSecondBest) {
					secondBestMatch = j;
					distanceSecondBest = tempDistance;
				}
		}
		if (bestMatch != -1) {
			if (secondBestMatch != -1) {
				PointPairRatio ppr;
				ppr.point1 = img1_features[i].point;
				ppr.bestMatch = img2_features[bestMatch].point;
				ppr.secondBestMatch = img2_features[secondBestMatch].point;
				ppr.ratio = distanceBest / distanceSecondBest;
				if (ppr.ratio <= RatioThreshold)
					pairs.push_back(ppr);
			}
			else
			{
				PointPairRatio ppr;
				ppr.point1 = img1_features[i].point;
				ppr.bestMatch = img2_features[bestMatch].point;
				ppr.secondBestMatch;
				ppr.ratio = RatioThreshold / 2;
			}
		}
	}
	return pairs;
}

void showMatches(cv::Mat &img1, cv::Mat &img2, cv::Mat &output, std::vector<PointPairDistance> &matches) {
	int columns = img1.cols + img2.cols;
	int rows = MAX(img1.rows, img2.rows);
	output.create(rows, columns, img1.type());
	img1.copyTo(output(cv::Rect(0, 0, img1.size().width, img1.size().height)));
	img2.copyTo(output(cv::Rect(img1.size().width, 0, img2.size().width, img2.size().height)));
	for (int i = 0; i < matches.size();i++) {
		cv::Scalar color(rand() % 256 / 255.0, rand() % 256 / 255.0, rand() % 256 / 255.0, 1);
		cv::line(output, cv::Point(matches[i].point1.y, matches[i].point1.x), cv::Point(matches[i].point2.y + img1.size().width, matches[i].point2.x), color, 1, 8, 0);
	}
//	cv::namedWindow("matches", cv::WINDOW_AUTOSIZE);
//	cv::imshow("matches", output);
}
void showMatches(cv::Mat &img1, cv::Mat &img2, cv::Mat &output, std::vector<PointPairRatio> &matches,bool invert) {
	std::vector<PointPairDistance> ppd(0);
	for (int i = 0; i < matches.size();i++) {
		PointPairDistance pair;
		if (!invert) {
			pair.point1 = matches[i].point1;
			pair.point2 = matches[i].bestMatch;
		}
		else
		{
			pair.point1 = cv::Point(matches[i].point1.y,matches[i].point1.x);
			pair.point2 = cv::Point(matches[i].bestMatch.y,matches[i].bestMatch.x);
		}
		pair.SSDdistance = 0;
		ppd.push_back(pair);
	}
	showMatches(img1, img2, output, ppd);
}
