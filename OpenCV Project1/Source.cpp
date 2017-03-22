#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>
#include <string>
#include <omp.h>
#include <iostream>

#include "FeatureMatcher.h"

#define inlierTreshold 10

#define delta 0.0001

// used to swap point coordinates
void inverPoint(cv::Point &point) {
	float x;
	x = point.x;
	point.x = point.y;
	point.y = x;
}
// swap all point coordinates in Point pair ratio (because open cv can be dumb when it comes to accessing memory, .at(a,b) == .at(Point(b,a)) 
void flipPointPair(PointPairRatio &r) {
	inverPoint(r.point1);
	inverPoint(r.bestMatch);
}

// returns a point obtained by projecting x and y by H, aswell as write this to x2 and y2
cv::Point project(float x1, float y1, cv::Mat H, float &x2, float &y2) {
	cv::Mat v1 = (cv::Mat_<double>(3, 1) << (double)x1, (double)y1, 1.0);
	cv::Mat result = H*v1;

	x2 = result.at<double>(0);
	y2 = result.at<double>(1);
	return cv::Point(x2, y2);
}
// returns a point obtained by projecting x and y by H
cv::Point project(float x,float y, cv::Mat H) {
	cv::Mat v1 = (cv::Mat_<double>(3, 1) << (double)x, (double)y, 1.0);
	cv::Mat result = H*v1;

	return cv::Point(result.at<double>(0), result.at<double>(1));
}
// coutns number of inliners based on a homography
std::vector<PointPairRatio> computeInlierCount(cv::Mat &H, std::vector<PointPairRatio> &matches, int numMatches, float inlierThreshold) {
	std::vector<PointPairRatio> resultVector(0);

	for (int i = 0; i < matches.size();i++) {

		float xResult, yResult;
		project((float)matches.at(i).point1.x, (double)matches.at(i).point1.y, H, xResult, yResult);
		cv::Point2f p1(xResult, yResult);
		float x = (p1.x - matches.at(i).bestMatch.x);
		x *= x;
		float y= (p1.y - matches.at(i).bestMatch.y);
		y *= y;

		float dist = sqrt(x + y);
		
		if (dist < inlierThreshold) {
			resultVector.push_back(matches.at(i));

		}
	}
	return resultVector;
}

// ransac function, returns same result as cv::findHomography wiht ransac enabled
bool RANSAC(std::vector<PointPairRatio> &matches, int numMatches, int numIterations,float  inlierThreshold, cv::Mat &hom, cv::Mat &homInv, cv::Mat &image1Display, cv::Mat &image2Display,bool saveImage,std::string saveName) {
	cv::Mat H;
	int numInliers = 0;
	bool hset = false;


	// break up PointPair ratio into 2 vectors which hole point and corresponding destination
	std::vector<cv::Point2f> sourcePoints(matches.size());
	std::vector<cv::Point2f> destinationPoints(matches.size());
	for (int i = 0; i < matches.size();i++) {
		sourcePoints.at(i) = cv::Point2f(matches.at(i).point1);
		destinationPoints.at(i) = cv::Point2f(matches.at(i).bestMatch);
	}
	//randomly select 4 different integeres within range 
	for (int i = 0; i < numIterations;i++) {

		int indicies[4];

		for (int x = 0; x < 4;x++) {
			bool set = false;
			while (!set) {
				indicies[x] = rand() % matches.size();
				set = true;
				for (int y = 0; y < x;y++) {
					if (indicies[y] == indicies[x]) {
						set = false;
						break;
					}
				}
			}
		}
		// create temporary point vecotrs who's data is data at 4 random positions within vectors
		std::vector<cv::Point2f> tempSource(4), tempDestination(4);

		for (int x = 0; x < 4;x++) {
			tempSource.at(x) = sourcePoints.at(indicies[x]);
			tempDestination.at(x) = destinationPoints.at(indicies[x]);
		}
		//find temp homography and store inlier number in tempNumInliers
		cv::Mat tempH = cv::findHomography(tempSource, tempDestination, 0);
		int tempNumInliers = computeInlierCount(tempH, matches, matches.size(), inlierThreshold).size();

		//check if tempH has more inliers than current best, if so store tempH as best
		if (tempNumInliers > numInliers) {
			H = tempH;
			numInliers = tempNumInliers;
			hset = true;
		}
	}

	// if we have set a homography at least once, then get new homography based on all inliers within current best homography
	if (hset) {
		std::vector<PointPairRatio> inliers = computeInlierCount(H, matches, matches.size(), inlierThreshold);

		std::vector<cv::Point2f> finalSourcePoints(inliers.size());
		std::vector<cv::Point2f> finalDestinationPoints(inliers.size());
		for (int i = 0; i < inliers.size();i++) {
			finalSourcePoints.at(i) = cv::Point2f(inliers.at(i).point1);
			finalDestinationPoints.at(i) = cv::Point2f(inliers.at(i).bestMatch);
		}
		hom = findHomography(finalSourcePoints, finalDestinationPoints, 0);
		homInv = hom.inv();


		//display
		cv::Mat output;
		cv::Mat saveFile;
		if (saveImage) {
			
			showMatches(image1Display, image2Display, output, inliers, true);
			output *= 255;

			output.convertTo(saveFile, CV_8UC3);

			cv::imwrite(saveName, saveFile);
		}
	}
	// no homography found;
	else {
		std::cout << "H NOT FOUND" << std::endl;
	}

	return hset;

}

// algorithm to blend images together
cv::Mat mergeImages(cv::Mat &img1, cv::Mat &img2, int cols,int rows) {
	cv::Mat output;
	output.create(rows, cols, img1.type());

	cv::Mat img1Channels[3];
	cv::Mat img2Channels[3];
	cv::Mat outChannels[3];

	cv::split(img1, img1Channels);
	cv::split(img2, img2Channels);
	cv::split(output, outChannels);
	for (int x = 0; x < rows;x++) {
		for (int y = 0; y < cols;y++) {
			float sum1, sum2;

			if (x < img1.rows && y < img1.cols) {
				sum1 = img1Channels[0].at<float>(x, y) + img1Channels[1].at<float>(x, y) + img1Channels[2].at<float>(x, y);
			}
			else
				sum1 = -1;
			if (x < img2.rows && y < img2.cols) {
				sum2 = img2Channels[0].at<float>(x, y) + img2Channels[1].at<float>(x, y) + img2Channels[2].at<float>(x, y);
			}
			else
				sum2 = -1;
			if (sum1 <delta && sum2 <delta) {
				for (int c = 0; c < 3;c++) {
					outChannels[c].at<float>(x, y) = 0;
				}
			}
			else if (sum1 <delta) {
				for (int c = 0; c < 3;c++) {
					outChannels[c].at<float>(x, y) = img2Channels[c].at<float>(x,y);
				}
			}
			else if (sum2 <delta) {
				for (int c = 0; c < 3;c++) {
					outChannels[c].at<float>(x, y) = img1Channels[c].at<float>(x, y);
				}
			}
			else  {
				for (int c = 0; c < 3;c++) {
					outChannels[c].at<float>(x, y) =( img1Channels[c].at<float>(x, y)+ img2Channels[c].at<float>(x, y))*0.5f; //(img1Channels[c].at<float>(x, y) + img2Channels[c].at<float>(x, y))*0.5f;
			

				}
			}


		}
	}


	cv::merge(outChannels, 3, output);
	return output;
}
// stitch im2 onto image 1 based on homography
void stitch(cv::Mat &img1, cv::Mat &img2, cv::Mat &hom, cv::Mat &homInv, cv::Mat &stitchedImage) {
	cv::Point corners[8];
	corners[0] = project(0, 0, homInv);
	corners[1] = project(img2.rows,0,homInv);
	corners[2] = project( img2.rows,img2.cols ,homInv);
	corners[3] = project(0, img2.cols, homInv);

	corners[4] = cv::Point(0, 0);
	corners[5] = cv::Point(img1.rows,img1.cols);
	corners[6] = cv::Point(img1.rows,0);
	corners[7] = cv::Point(0,img1.cols);

	float minX = corners[0].y;
	float maxX = corners[0].y;
	float minY = corners[0].x;
	float maxY = corners[0].x;

	for (int i = 0; i < 8;i++) {
		minX = MIN(minX, corners[i].x);
		minY = MIN(minY, corners[i].y);

		maxX = MAX(maxX, corners[i].x);
		maxY = MAX(maxY, corners[i].y);

	}
	int height = maxX - 2*minX+200;
	int width = maxY - 2*minY+200;


	cv::Mat transformOffset = (cv::Mat_<double>(3, 3) <<
		1, 0, -2*minX,
		0, 1, -2*minY,
		0, 0, 1);



	cv::Mat img1Transform, img2Transform;
	cv::warpPerspective(img2, img2Transform, transformOffset*homInv, cv::Size(width, height));
	cv::warpPerspective(img1, img1Transform, transformOffset, cv::Size(width, height));
	stitchedImage = mergeImages(img1Transform, img2Transform, width,height);
}
 // trims images to be only box around extremidies
void trimImage(cv::Mat &src,cv::Mat &out) {
	int left, right, top, down;
	cv::Mat chan[3];
	cv::split(src, chan);
	int xfirst, xlast, yfirst, ylast;
	bool found = false;
	std::cout << "starting trim..." << std::endl;
	for (int i = 0; i < src.rows;i++) {
		for (int j = 0; j < src.cols;j++) {
			float sum = chan[0].at<float>(i, j) + chan[1].at<float>(i, j) + chan[2].at<float>(i, j);
			if (sum > delta) {
				top = i;
				found = true;
				break;
			}
		}
		if (found)
			break;
	}
	found = false;

	for (int i = src.rows-1; i >=0;i--) {
		for (int j = 0; j < src.cols;j++) {
			float sum = chan[0].at<float>(i, j) + chan[1].at<float>(i, j) + chan[2].at<float>(i, j);
			if (sum > delta) {
				down = i;
				found = true;
				break;
			}
		}
		if (found)
			break;
	}
	found = false;
	for (int j = 0; j < src.cols;j++) {
		for (int i = 0; i < src.rows;i++) {
			float sum = chan[0].at<float>(i, j) + chan[1].at<float>(i, j) + chan[2].at<float>(i, j);
			if (sum > delta) {
				left = j;
				found = true;
				break;
			}
		}
		if (found)
			break;
	}

	found = false;

	for (int j = src.cols-1; j >0;j--) {
		for (int i = 0; i < src.rows;i++) {
			float sum = chan[0].at<float>(i, j) + chan[1].at<float>(i, j) + chan[2].at<float>(i, j);
			if (sum > delta) {
				right = j;
				found = true;
				break;
			}
		}
		if (found)
			break;
	}
	
	std::cout << "Trimming: w: " << src.size().width << " h:" << src.size().height << std::endl;
	std::cout << "bounds: L: " << left << " R: " << right << " D: " << down << " T" << top << std::endl;
	out = src(cv::Rect(left,top,right-left,down-top));
}
// creates panorama between two images and saves to saveName
bool makePanorama(std::string imagePath1, std::string imagePath2,std::string saveName,bool showImage,cv::Mat &out) {
	
	cv::Mat img1, img2, result, H, hom, invHom;
	std::vector<Feature> features1 = getFeatureVector(imagePath1, img1);
	std::vector<Feature> features2 = getFeatureVector(imagePath2, img2);

	std::vector<PointPairRatio> matches = getBestFeatureMatchesRatio(features1, features2);

	for (int i = 0; i < matches.size();i++) {
		flipPointPair(matches.at(i));
	}

	std::cout << "Total matches " << matches.size() << std::endl;
	if (matches.size() > 8 && RANSAC(matches, matches.size(), 1000, inlierTreshold, hom, invHom, img1, img2, showImage,saveName+"_Matches.png")) {
		stitch(img1, img2, hom, invHom, result);
		cv::Mat saveFile;
		result *= 255;
		result.convertTo(saveFile, CV_8UC3);
		result /= 255;

		cv::imwrite(saveName, saveFile);
		trimImage(result, out);
		return true;
	}
	else {
		cv::Mat saveFile;
		std::cout << "Not enough matches found";
		img1 *= 255;
		img1.convertTo(saveFile, CV_8UC3);
		img1 /= 255;
		cv::imwrite(saveName, saveFile);
		return false;


	}
}
int main() {
	std::string imagePath1 = "images/Rainier1.png";
	std::string imagePath2 = "images/Rainier2.png";
	std::string imagePath3 = "images/Rainier3.png";
	std::string imagePath4 = "images/Rainier4.png";
	std::string imagePath5 = "images/Rainier5.png";
	std::string imagePath6 = "images/Rainier6.png";
	std::string saveName1 = "images/output/RainierPano1.png";
	std::string saveName2 = "images/output/RainierPano2.png";
	std::string saveName3 = "images/output/RainierPano3.png";
	std::string saveName4 = "images/output/RainierPano4.png";

	std::string saveFinal = "images/output/finalPano.png";


	// stiching all parts of rainier together
	cv::Mat r1,r2,r3,r4,r5;
	bool s1, s2, s3, s4, s5;
	if (makePanorama(imagePath1, imagePath2, saveName1, true, r5)) {
		s1 = true;
	}
	
	std::cout << "Next image";
	if (makePanorama(imagePath3, imagePath4, saveName2, true, r5)) {
		s2 = true;

	}

	std::cout << "Next image";

	if (makePanorama(imagePath5, imagePath6, saveName3, true, r5)) {
		s3 = true;

	}

	std::cout << "Next image";

	if (makePanorama(saveName1, saveName2,saveName4, true, r5)) {
		s4 = true;

	}
	std::cout << "Next image";

	if (makePanorama(saveName4, saveName3,saveFinal, true, r5)) {
		s5 = true;

	}

	if (s5)
	
	cv::imshow("res1: ", r5);
	
	
	cv::waitKey(0);

	std::cout << "Done";


}