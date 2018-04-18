#include <iostream>
#include <vector>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>

#include "linefinder.h"
#include "harrisDetector.h"

// Drawing optical flow vectors on an image
void drawOpticalFlow(const cv::Mat& oflow,    // the optical flow
	                 cv::Mat& flowImage,      // the produced image
	                 int stride,  // the stride for displaying the vectors
	                 float scale, // multiplying factor for the vectors
	                 const cv::Scalar& color) // the color of the vectors
{
	// create the image if required
	if (flowImage.size() != oflow.size()) {
		flowImage.create(oflow.size(), CV_8UC3);
		flowImage = cv::Vec3i(255,255,255);
	}

	// for all vectors using stride as a step
	for (int y = 0; y < oflow.rows; y += stride)
		for (int x = 0; x < oflow.cols; x += stride) {
			// gets the vector
			cv::Point2f vector = oflow.at< cv::Point2f>(y, x);
			// draw the line
			cv::line(flowImage, cv::Point(x, y),
				     cv::Point(static_cast<int>(x + scale*vector.x + 0.5),
						       static_cast<int>(y + scale*vector.y + 0.5)), color);
			// draw the arrow tip
			cv::circle(flowImage, cv::Point(static_cast<int>(x + scale*vector.x + 0.5),
				                            static_cast<int>(y + scale*vector.y + 0.5)), 1, color, -1);
		}
}

double computeVectorLength(cv::Point point1, cv::Point point2){
	return sqrt(pow(point1.x-point2.x, 2) + pow(point1.y-point2.y, 2));
}

int main(){
	// search in all 81 images two at a time to compare the median flowImage
	//for this reason ww will start at image 2 and the last one read will be image 81
	for (int i = 2; i < 82; i++){
	  // read the first image as a color image
		cv::Mat image0;
		cv::Mat image1;
		if(i<10) {
		image0 = cv::imread("images/000000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/000000000"+std::to_string(i-1)+".jpg", 0);
		/*image0 = cv::imread("images/video/video2/000000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video2/000000000"+std::to_string(i-1)+".jpg", 0);*/
		/*image0 = cv::imread("images/video/video6/000000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video6/000000000"+std::to_string(i-1)+".jpg", 0);*/
			/*image0 = cv::imread("images/video/video7/000000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video7/000000000"+std::to_string(i-1)+".jpg", 0);*/
		}
		else if (i ==  10)
		{
		image0 = cv::imread("images/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/000000000"+std::to_string(i-1)+".jpg", 0);
		/*image0 = cv::imread("images/video/video2/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video2/000000000"+std::to_string(i-1)+".jpg", 0);*/
		/*image0 = cv::imread("images/video/video6/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video6/000000000"+std::to_string(i-1)+".jpg", 0);*/
			/*image0 = cv::imread("images/video/video7/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video7/000000000"+std::to_string(i-1)+".jpg", 0);*/
		}
		else {
		image0 = cv::imread("images/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/00000000"+std::to_string(i-1)+".jpg", 0);
		 /*image0 = cv::imread("images/video/video2/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video2/00000000"+std::to_string(i-1)+".jpg", 0);*/
			/*image0 = cv::imread("images/video/video6/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video6/00000000"+std::to_string(i-1)+".jpg", 0);*/
			/*image0 = cv::imread("images/video/video7/00000000"+std::to_string(i)+".jpg", 0);
			image1 = cv::imread("images/video/video7/00000000"+std::to_string(i-1)+".jpg", 0);*/
		}
		if (!image0.data || !image1.data) return 0;

		//apply gaussian filter (low pass filter) to reduce noise
		//cv::GaussianBlur(image0, image0, cv::Size(3, 3), 1.5);

		// define image ROI around front of camera
	  int xo=220, yo=180;
		int width=200, height=200;
		/*int xo=250, yo=150;
		int width=120, height=250;*/
		/*int xo=280, yo=50;
		int width=100, height=200;*/
		/*int xo=220, yo=100;
		int width=200, height=280;*/

		// draw a line from where we will start computing the sobel
		cv::line(image0,cv::Point(xo,yo+height),cv::Point(xo+width,yo+height),cv::Scalar(255),2);
		cv::namedWindow("Initial Image");
		cv::imshow("Initial Image",image0);

		//create ROI on which we are going to do all the image processing
		cv::Rect r(xo,yo,width,height);
		cv::rectangle(image0,r,0);
		cv::Mat roi0(image0(r));
		cv::Mat roi1(image1(r));



		cv::Mat grad_y;
		int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

		//sobel y
		Sobel( roi0, grad_y, CV_8U, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
		cv::namedWindow("SOBEL Y Contours");
		cv::imshow("SOBEL Y Contours",grad_y);

		// Apply Canny algorithm
		cv::Mat contours;
		cv::Canny(grad_y,contours,125,350);

		// Display the image of contours
		//cv::namedWindow("Canny Contours");
		//cv::imshow("Canny Contours",255-contours);

		/***************START OF INTEREST POINTS***********************/
		// GFTT:
		// Compute good features to track

		//std::vector<cv::KeyPoint> keypoints;
		// GFTT detector
		/*cv::Ptr<cv::GFTTDetector> ptrGFTT = cv::GFTTDetector::create(
			500,	// maximum number of keypoints to be returned
			0.01,	// quality level
			10);	// minimum allowed distance between points
		// detect the GFTT
		ptrGFTT->detect(roi0,keypoints);
		// for all keypoints
		std::vector<cv::KeyPoint>::const_iterator iter= keypoints.begin();
		while (iter!=keypoints.end()) {

			// draw a circle at each corner location
			cv::circle(roi0,iter->pt,3,cv::Scalar(255,255,255),1);
			++iter;
		}

	    // Display the keypoints
		cv::namedWindow("GFTT");
		cv::imshow("GFTT",roi0);

		// draw the keypoints
		cv::drawKeypoints(roi0,		// original image
			keypoints,					// vector of keypoints
			roi0,						// the resulting image
			cv::Scalar(255,255,255),	// color of the points
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

	    // Display the keypoints
		cv::namedWindow("Good Features to Track Detector");
		cv::imshow("Good Features to Track Detector",roi0);*/

		// FAST feature:


		/*keypoints.clear();
		// FAST detector
		cv::Ptr<cv::FastFeatureDetector> ptrFAST = cv::FastFeatureDetector::create(40);
		// detect the keypoints
		ptrFAST->detect(roi0,keypoints);
		// draw the keypoints
		cv::drawKeypoints(roi0,keypoints,roi0,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
		std::cout << "Number of keypoints (FAST): " << keypoints.size() << std::endl;

	    // Display the keypoints
		cv::namedWindow("FAST");
		cv::imshow("FAST",roi0);*/

		// FAST feature without non-max suppression

		/*keypoints.clear();
		// detect the keypoints
		ptrFAST->setNonmaxSuppression(false);

		ptrFAST->detect(roi0, keypoints);
		// draw the keypoints
		cv::drawKeypoints(roi0,keypoints,roi0,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

	    // Display the keypoints
		cv::namedWindow("FAST Features (all)");
		cv::imshow("FAST Features (all)",roi0);*/


		// Combined display
		/*cv::Mat combined(image1.rows, image1.cols + image0.cols, CV_8U);
		image1.copyTo(combined.colRange(0, image1.cols));
		image0.copyTo(combined.colRange(image1.cols, image1.cols+image0.cols));
		cv::imshow("Frames", combined);*/

		// Create the optical flow algorithm
		cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();

		/*std::cout << "regularization coeeficient: " << tvl1->getLambda() << std::endl; // the smaller the soomther
		std::cout << "Number of scales: " << tvl1->getScalesNumber() << std::endl; // number of scales
		std::cout << "Scale step: " << tvl1->getScaleStep() << std::endl; // size between scales
		std::cout << "Number of warpings: " << tvl1->getWarpingsNumber() << std::endl; // size between scales
		std::cout << "Stopping criteria: " << tvl1->getEpsilon() << " and " << tvl1->getOuterIterations() << std::endl; // size between scales
		*/																												// compute the optical flow between 2 frames
		cv::Mat oflow; // image of 2D flow vectors
		// compute optical flow between frame1 and frame2
		tvl1->setLambda(0.075);
		tvl1->calc(roi1, roi0, oflow);

		// Draw the optical flow image
		cv::Mat flowImage;
		drawOpticalFlow(oflow,     // input flow vectors
			roi0, // image to be generated
			8,         // display vectors every 8 pixels
			2,         // multiply size of vectors by 2
			cv::Scalar(0, 0, 0)); // vector color

		//cv::imshow("Optical Flow", image0);

		/****************END OF INTEREST POINTS***********************/

		// Hough tranform for line detection
		std::vector<cv::Vec2f> lines;
		//cv::HoughLines(contours,lines,1,PI/180,30);
		cv::HoughLines(contours,lines,1,PI/180,30);

		// Draw the lines
		cv::Mat result(contours.rows,contours.cols,CV_8U,cv::Scalar(255));
		roi0.copyTo(result);

		//std::cout << "Lines detected: " << lines.size() << std::endl;

		std::vector<cv::Vec2f>::const_iterator it= lines.begin();
		while (it!=lines.end()) {

			float rho= (*it)[0];   // first element is distance rho
			float theta= (*it)[1]; // second element is angle theta

			if (theta < PI/4. || theta > 3.*PI/4.) { // ~vertical line

				// point of intersection of the line with first row
				cv::Point pt1(rho/cos(theta),0);
				// point of intersection of the line with last row
				cv::Point pt2((rho-result.rows*sin(theta))/cos(theta),result.rows);
				// draw a white line
				cv::line( result, pt1, pt2, cv::Scalar(255), 1);

			} else { // ~horizontal line

				// point of intersection of the line with first column
				cv::Point pt1(0,rho/sin(theta));
				// point of intersection of the line with last column
				cv::Point pt2(result.cols,(rho-result.cols*cos(theta))/sin(theta));
				// draw a white line
				cv::line( result, pt1, pt2, cv::Scalar(255), 1);
			}

			//std::cout << "line: (" << rho << "," << theta << ")\n";

			++it;
		}

		// Display the detected line image
		//cv::namedWindow("Lines with Hough");
		//cv::imshow("Lines with Hough",result);

		// Create LineFinder instance
		LineFinder ld;

		// Set probabilistic Hough parameters
		ld.setLineLengthAndGap(90,20);
		ld.setMinVote(50);

		// Detect lines
		std::vector<cv::Vec4i> li= ld.findLines(contours);
		ld.drawDetectedLines(roi0);
		cv::namedWindow("Lines with HoughP");
		cv::imshow("Lines with HoughP",image0);

		std::vector<cv::Vec4i>::const_iterator it2= li.begin();
		/*while (it2!=li.end()) {

			std::cout << "li :(" << (*it2)[0] << ","<< (*it2)[1]<< ")-("
				     << (*it2)[2]<< "," << (*it2)[3] << ")" <<std::endl;

			++it2;
		}*/

		//we need to compute the distances between consecutive lines in li
		//and to select the band of horizontal lines closest to each other
		//to detect a car

		//first we are going to pass through li and only keep lines whose
		//y point coordinates difference is less than 5
		//li with only horizontal Lines
		std::vector<cv::Vec4i> liHorizontal;
		std::vector<cv::Vec4i>::const_iterator it_horizontal= li.begin();
		while (it_horizontal!=li.end()) {
			if( std::abs( (*it_horizontal)[1] - (*it_horizontal)[3] ) < 10){
				liHorizontal.push_back(*it_horizontal);
			}
			++it_horizontal;
		}

		std::vector<cv::Vec4i>::const_iterator it3= liHorizontal.begin();
		/*while (it3!=liHorizontal.end()) {

			std::cout << "li_horizontal: (" << (*it3)[0] << ","<< (*it3)[1]<< ")-("
				     << (*it3)[2]<< "," << (*it3)[3] << ")" <<std::endl;

			++it3;
		}*/
		//sorting the vector to have the lines in order
		std::sort(liHorizontal.begin(), liHorizontal.end(), [](const cv::Vec4i &left, const cv::Vec4i &right) {
		    return left[1] < right[1];
		});

		std::vector<cv::Vec4i>::const_iterator itsort= liHorizontal.begin();
		/*while (itsort!=liHorizontal.end()) {

			std::cout << "li_horizontal_sort: (" << (*itsort)[0] << ","<< (*itsort)[1]<< ")-("
				     << (*itsort)[2]<< "," << (*itsort)[3] << ")" <<std::endl;

			++itsort;
		}*/

		//reverse the vector to have the lines starting from bottom to top
		std::reverse(liHorizontal.begin(),liHorizontal.end());

		std::vector<cv::Vec4i>::const_iterator itreverse= liHorizontal.begin();
		/*while (itreverse!=liHorizontal.end()) {

			std::cout << "li_horizontal_reverse: (" << (*itreverse)[0] << ","<< (*itreverse)[1]<< ")-("
				     << (*itreverse)[2]<< "," << (*itreverse)[3] << ")" <<std::endl;

			++itreverse;
		}*/

		//pass through the vector to progressively identify the band of lines closest together
		std::vector<cv::Vec4i>::const_iterator it4= liHorizontal.begin();
		std::vector<cv::Vec4i>::const_iterator it5;

		//int number of consecutive lines to accept it as a car border
		int trigger = 3;
		int consecutiveLines;
		bool terminateOuterLoop = false;
		cv::Vec4i carBorderLine;
		//the first carBorderLine from the bottom is the one returned
		while (it4!=liHorizontal.end() && !terminateOuterLoop) {
			it5 = it4 + 1;
			consecutiveLines = 0;
			//std::cout << "1st while" << std::endl;
			while (it5!=liHorizontal.end()) {
				//distance between lines is less than
				//std::cout << "2nd while" << std::endl;
				if (std::abs((*it4)[1] - (*it5)[1]) < 50 && consecutiveLines < trigger){
					++consecutiveLines;
					//std::cout << "increment consecutiveLines = " << consecutiveLines << std::endl;
					if(consecutiveLines == trigger){
						//std::cout << "trigger reached " << std::endl;
						carBorderLine = *it4;
						terminateOuterLoop = true;
						break;
					}
					++it5;
				} else {
					//the 2 lines are too apart we stop
					break;
				}

			}
			++it4;
		}

		//std::cout << "carBorderLine: (" << carBorderLine[0] << ","<< carBorderLine[1]<< ")-("
		//			 << carBorderLine[2]<< "," << carBorderLine[3] << ")" <<std::endl;

		//image0 = cv::imread("images/0000000000.jpg",0);
		//cv::Mat roiCarBumper(image0, cv::Rect(xo,yo,width,height));

		//reload image to separately display the computing of all the parameters
		if(i<10) image0 = cv::imread("images/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		/*if(i<10) image0 = cv::imread("images/video/video2/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video2/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/
		/*if(i<10) image0 = cv::imread("images/video/video6/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video6/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/
		/*if(i<10) image0 = cv::imread("images/video/video7/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video7/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/

		cv::Mat roiCarBumper(image0, cv::Rect(xo,yo,width,height));




		/*COLLISION DETECTION*/
		//aT this point try to compute the total number of arrows with a y coordinates
		//close by x factor to the line of end of the car
		// if we have let's say more than half of arrows at that level in comparison
		//to the total number of possible arrows (8 pixels of step and 200 pixels of height)
		//we can assume that the general direction is downwards toward the car and draw a red lines
		//to indicate this to the driverless car


		int stride = 8;
		int scaleFactor = 2;
		int counterTowards = 0;
		int counterAway = 0;
		cv::Scalar color = cv::Scalar(0, 0, 0);
		cv::Scalar colorDangerous = cv::Scalar(0, 0, 255);
		cv::Scalar colorGood = cv::Scalar(0, 255, 0);
		cv::Point2f point1, point2; //start and end of median flow arrow

		cv::Point2f towardsVectorPoint1 = cv::Point(0, 0);
		cv::Point2f towardsVectorPoint2 = cv::Point(0, 0);
		cv::Point2f awayVectorPoint1 = cv::Point(0, 0);
		cv::Point2f awayVectorPoint2 = cv::Point(0, 0);

		// for all vectors using stride as a step
		for (int y = 0; y < oflow.rows; y += stride)
			for (int x = 0; x < oflow.cols; x += stride) {
				// gets the vector
				cv::Point2f vector = oflow.at< cv::Point2f>(y, x);
				point1 = cv::Point(x, y);
				point2 = cv::Point(static_cast<int>(x + scaleFactor*vector.x + 0.5),
									static_cast<int>(y + scaleFactor*vector.y + 0.5));

				//for each median flow vector we will first retrieve all the arrows that are pointing downwards
				//and for all this downwards arrows we will retain the ones that are in the vicinity of the carbumper Y coordinates
				if ( (point2.y > point1.y) &&
							computeVectorLength(point1, point2) > 3 &&
						 (point2.x >= carBorderLine[0]) && (point2.x <= carBorderLine[2]) &&
					   (std::abs(point2.y - carBorderLine[1]) < 50 || std::abs(point2.y - carBorderLine[3]) < 50)
					 ){
						std::cout << "Dangerous Line detected: (" << point1.x << ", "<< point1.y << ")-("
						  			  << point2.x<< "," << point2.y << ")" <<std::endl;
						std::cout << "carBorderLine: (" << carBorderLine[0] << ","<< carBorderLine[1]<< ")-("
						  			  << carBorderLine[2]<< "," << carBorderLine[3] << ")" <<std::endl;

						// draw the line
						cv::line(roiCarBumper,
										 point1,
							       point2,
										 colorDangerous);
						// draw the arrow tip
						cv::circle(roiCarBumper,
							         point2,
											 1,
											 colorDangerous,
											 -1);

						//updating the towards resulting vector
						towardsVectorPoint1.x += point1.x;
						towardsVectorPoint1.y += point1.y;
						towardsVectorPoint2.x += point2.x;
						towardsVectorPoint2.y += point2.y;
						counterTowards++;
					} else if ( (point2.y < point1.y) &&
											computeVectorLength(point1, point2) > 3 &&
											(point2.x >= carBorderLine[0]) && (point2.x <= carBorderLine[2]) &&
											(std::abs(point2.y - carBorderLine[1]) < 50 || std::abs(point2.y - carBorderLine[3]) < 50)
					){
						std::cout << "Good Line detected: (" << point1.x << ", "<< point1.y << ")-("
						  			  << point2.x<< "," << point2.y << ")" <<std::endl;
						std::cout << "carBorderLine: (" << carBorderLine[0] << ","<< carBorderLine[1]<< ")-("
						  			  << carBorderLine[2]<< "," << carBorderLine[3] << ")" <<std::endl;

						// draw the line
						cv::line(roiCarBumper,
										 point1,
							       point2,
										 colorGood);
						// draw the arrow tip
						cv::circle(roiCarBumper,
							         point2,
											 1,
											 colorGood,
											 -1);

						//updating the away resulting vector
 						awayVectorPoint1.x += point1.x;
 						awayVectorPoint1.y += point1.y;
 						awayVectorPoint2.x += point2.x;
 						awayVectorPoint2.y += point2.y;
 						counterAway++;
					} else {
						// draw the line
						cv::line(roiCarBumper,
										 point1,
										 point2,
										 color);
						// draw the arrow tip
						cv::circle(roiCarBumper,
											 point2,
											 1,
											 color,
											 -1);
					}

			}

		cv::rectangle(image0, cv::Rect(xo,yo,width,height), cv::Scalar(255),1);
		cv::line(roiCarBumper, cv::Point(carBorderLine[0],carBorderLine[1]),cv::Point(carBorderLine[2],carBorderLine[3]),cv::Scalar(255),3);

		cv::namedWindow("Front Car Bumper Detection");
		cv::imshow("Front Car Bumper Detection",image0);

		towardsVectorPoint1.x /= counterTowards;
		towardsVectorPoint1.y /= counterTowards;
		towardsVectorPoint1.y -= 25;
		towardsVectorPoint2.x /= counterTowards;
		towardsVectorPoint2.y /= counterTowards;

		awayVectorPoint1.x /= counterAway;
		awayVectorPoint1.y /= counterAway;
		awayVectorPoint1.y += 25;
		awayVectorPoint2.x /= counterAway;
		awayVectorPoint2.y /= counterAway;

		if(i<10) image0 = cv::imread("images/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		/*if(i<10) image0 = cv::imread("images/video/video2/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video2/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/
		/*if(i<10) image0 = cv::imread("images/video/video6/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video6/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/
		/*if(i<10) image0 = cv::imread("images/video/video7/000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else image0 = cv::imread("images/video/video7/00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);*/

		cv::Mat roiFinalresult(image0, cv::Rect(xo,yo,width,height));

		// draw the resulting towards vector line
		cv::arrowedLine(roiFinalresult,
						 towardsVectorPoint1,
						 towardsVectorPoint2,
						 colorDangerous,
					   2,
					 	 8,
					   0,
					   0.5);

		/* draw the arrow tip
 		/cv::circle(roiFinalresult,
 							 towardsVectorPoint2,
 							 1,
 							 colorDangerous,
 							 3);*/

	  // draw the resulting away vector line
		cv::arrowedLine(roiFinalresult,
						 awayVectorPoint1,
						 awayVectorPoint2,
						 colorGood,
					   2,
						 8,
					   0,
					   0.5);

	 	// draw the arrow tip
		/*cv::circle(roiFinalresult,
							 awayVectorPoint2,
							 1,
							 colorGood,
							 3);*/

		cv::namedWindow("Collision detector");
		cv::imshow("Collision detector",image0);

		cv::waitKey(100);
	}

  cv::waitKey();
}
