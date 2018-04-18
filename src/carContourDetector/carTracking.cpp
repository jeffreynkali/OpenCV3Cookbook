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

#define VIDEO1_FOLDER "images/video/video1/"
#define VIDEO1_NB_IMG 82
#define VIDEO2_FOLDER "images/video/video2/"
#define VIDEO2_NB_IMG 91
#define VIDEO6_FOLDER "images/video/video6/"
#define VIDEO6_NB_IMG 72
#define VIDEO7_FOLDER "images/video/video7/"
#define VIDEO7_NB_IMG 92
#define VIDEO10_FOLDER "images/video/video10/"
#define VIDEO10_NB_IMG 113
#define VIDEO11_FOLDER "images/video/video11/"
#define VIDEO11_NB_IMG 89
// define image ROI around front of camera
#define VIDEO1_X0 230
#define VIDEO1_Y0 170
#define VIDEO1_WIDTH 170
#define VIDEO1_HEIGHT 220
#define VIDEO1_HOUGHP_MIN_LINE_LENGTH 90
#define VIDEO1_HOUGHP_MAX_LINE_GAP 20
#define VIDEO1_HOUGHP_MIN_VOTE 50
#define VIDEO2_X0 230
#define VIDEO2_Y0 100
#define VIDEO2_WIDTH 140
#define VIDEO2_HEIGHT 270
#define VIDEO2_HOUGHP_MIN_LINE_LENGTH 10
#define VIDEO2_HOUGHP_MAX_LINE_GAP 50
#define VIDEO2_HOUGHP_MIN_VOTE 15
#define VIDEO6_X0 280
#define VIDEO6_Y0 50
#define VIDEO6_WIDTH 100
#define VIDEO6_HEIGHT 250
#define VIDEO6_HOUGHP_MIN_LINE_LENGTH 20
#define VIDEO6_HOUGHP_MAX_LINE_GAP 20
#define VIDEO6_HOUGHP_MIN_VOTE 20
#define VIDEO7_X0 200
#define VIDEO7_Y0 100
#define VIDEO7_WIDTH 150
#define VIDEO7_HEIGHT 280
#define VIDEO7_HOUGHP_MIN_LINE_LENGTH 20
#define VIDEO7_HOUGHP_MAX_LINE_GAP 20
#define VIDEO7_HOUGHP_MIN_VOTE 25
#define VIDEO10_X0 230
#define VIDEO10_Y0 50
#define VIDEO10_WIDTH 150
#define VIDEO10_HEIGHT 280
#define VIDEO10_HOUGHP_MIN_LINE_LENGTH 20
#define VIDEO10_HOUGHP_MAX_LINE_GAP 20
#define VIDEO10_HOUGHP_MIN_VOTE 25
#define VIDEO11_X0 270
#define VIDEO11_Y0 140
#define VIDEO11_WIDTH 120
#define VIDEO11_HEIGHT 240
#define VIDEO11_HOUGHP_MIN_LINE_LENGTH 20
#define VIDEO11_HOUGHP_MAX_LINE_GAP 20
#define VIDEO11_HOUGHP_MIN_VOTE 20

using namespace std;
using namespace cv;

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

// Given two points compute the length of the distance vector between them
double computeVectorLength(cv::Point point1, cv::Point point2){
	return sqrt(pow(point1.x-point2.x, 2) + pow(point1.y-point2.y, 2));
}

int main(){

	/*1 6 and 11 are working fine 7 is ok from the middle of the video still more work for 2 and 10*/

	//video configuration to use
	string video_folder =  VIDEO1_FOLDER;
	int video_nb_img = VIDEO1_NB_IMG;
	int video_x0 = VIDEO1_X0;
	int video_y0 = VIDEO1_Y0;
	int video_width = VIDEO1_WIDTH;
	int video_height = VIDEO1_HEIGHT;
	int minlinelength = VIDEO1_HOUGHP_MIN_LINE_LENGTH;
	int maxlinegap = VIDEO1_HOUGHP_MAX_LINE_GAP;
	int minvote = VIDEO1_HOUGHP_MIN_VOTE;

	// search in all images two at a time to compare the median flowImage
	for (int i = 1; i < video_nb_img; i++){
	  // read the first image as a color image
		cv::Mat image1;
		cv::Mat image2;

		if (i<10) {
		    image1 = cv::imread(std::string(video_folder) + "000000000"+std::to_string(i-1)+".jpg", 0);
		    image2 = cv::imread(std::string(video_folder) + "000000000"+std::to_string(i)+".jpg", 0);
		} else if (i ==  10) {
		    image1 = cv::imread(std::string(video_folder) + "000000000"+std::to_string(i-1)+".jpg", 0);
		    image2 = cv::imread(std::string(video_folder) + "00000000"+std::to_string(i)+".jpg", 0);
		} else if (i > 10 && i < 100) {
		    image1 = cv::imread(std::string(video_folder) + "00000000"+std::to_string(i-1)+".jpg", 0);
		    image2 = cv::imread(std::string(video_folder) + "00000000"+std::to_string(i)+".jpg", 0);
		} else if ( i == 100) {
        image1 = cv::imread(std::string(video_folder) + "00000000"+std::to_string(i-1)+".jpg", 0);
        image2 = cv::imread(std::string(video_folder) + "0000000"+std::to_string(i)+".jpg", 0);
    } else {
        image1 = cv::imread(std::string(video_folder) + "0000000"+std::to_string(i-1)+".jpg", 0);
        image2 = cv::imread(std::string(video_folder) + "0000000"+std::to_string(i)+".jpg", 0);
    }

		if (!image1.data || !image2.data) {
			std::cout << "Something went wrong with the data for images 0 or 1" << std::endl;
			return 0;
		}

		//apply gaussian filter (low pass filter) to reduce noise
		//cv::GaussianBlur(image1, image1, cv::Size(3, 3), 1.5);

		// draw a line from where we will start computing the sobel
		//cv::line(image1,cv::Point(VIDEO1_X0,VIDEO1_Y0+VIDEO1_HEIGHT),cv::Point(VIDEO1_X0+VIDEO1_WIDTH,VIDEO1_Y0+VIDEO1_HEIGHT),cv::Scalar(255),2);
		//cv::namedWindow("Initial Image");
		//cv::imshow("Initial Image",image1);

		//create ROI on which we are going to do all the image processing
		cv::Rect r(video_x0,video_y0,video_width,video_height);
		cv::rectangle(image1,r,0);
    cv::rectangle(image2,r,0);
		cv::Mat roi1(image1(r));
		cv::Mat roi2(image2(r));

		cv::Mat grad_y;
		int scale = 1;
  	int delta = 0;
  	int ddepth = CV_16S;

		//sobel y
		//Sobel( roi1, grad_y, CV_8U, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

		//cv::namedWindow("SOBEL Y Contours");
		//cv::imshow("SOBEL Y Contours",grad_y);

		// Apply Canny algorithm
		// Note that Canny algorithm uses Sobel in the background
		cv::Mat contours;
		cv::Mat canny_output;
		vector<vector<Point> > contourVectors;
		vector<Point> longestContour;
		int maxContourLength = 0;
  	vector<Vec4i> hierarchy;
		RNG rng(12345);
		//cv::Canny(grad_y,contours,125,350);
		cv::Canny(roi1,canny_output,125,350);

		/// Find contours
    findContours( canny_output, contourVectors, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
		/// Draw contours
    cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
	  for( int i = 0; i< contourVectors.size(); i++ )
	     {
	       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
				 //std::cout << "Length of vector of points contourVectors[" << i << "]: " << contourVectors[i].size() << ", " <<std::endl;
				 drawContours( drawing, contourVectors, i, color, 2, 8, hierarchy, 0, Point() );
	     }

	  /// Show in a window
	  //cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	  //cv::imshow( "Contours", drawing );
		canny_output.copyTo(contours);
		// Display the image of contours
		//cv::namedWindow("Canny Contours");
		//cv::imshow("Canny Contours",255-contours);

		// Create the optical flow algorithm
		cv::Ptr<cv::DualTVL1OpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();																											// compute the optical flow between 2 frames
		cv::Mat oflow; // image of 2D flow vectors
		// compute optical flow between frame1 and frame2
		tvl1->setLambda(0.075);
		tvl1->calc(roi1, roi2, oflow);

		// Draw the optical flow image
		cv::Mat flowImage;
		drawOpticalFlow(oflow,     // input flow vectors
			roi2, // image to be generated
			8,         // display vectors every 8 pixels
			2,         // multiply size of vectors by 2
			cv::Scalar(0, 0, 0)); // vector color

		//cv::imshow("Optical Flow", image1);

		// Create LineFinder instance
		LineFinder ld;

		// Set probabilistic Hough parameters
		ld.setLineLengthAndGap(minlinelength,maxlinegap);
		ld.setMinVote(minvote);

		// Detect lines
		std::vector<cv::Vec4i> li= ld.findLines(contours);
		ld.drawDetectedLines(roi1);
		cv::namedWindow("Lines with HoughP");
		cv::imshow("Lines with HoughP",image1);

		std::vector<cv::Vec4i>::const_iterator it2= li.begin();
		//we need to compute the distances between consecutive lines in li
		//and to select the band of horizontal lines closest to each other
		//to detect a car

		//first we are going to pass through li and only keep lines whose
		//y point coordinates difference is less than 10
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

		//sorting the vector to have the lines in order
		std::sort(liHorizontal.begin(), liHorizontal.end(), [](const cv::Vec4i &left, const cv::Vec4i &right) {
		    return left[1] < right[1];
		});

		std::vector<cv::Vec4i>::const_iterator itsort= liHorizontal.begin();

		//reverse the vector to have the lines starting from bottom to top
		std::reverse(liHorizontal.begin(),liHorizontal.end());

		std::vector<cv::Vec4i>::const_iterator itreverse= liHorizontal.begin();

		//pass through the vector to progressively identify the band of lines closest together
		std::vector<cv::Vec4i>::const_iterator it4= liHorizontal.begin();
		std::vector<cv::Vec4i>::const_iterator it5;

		//int number of consecutive lines to accept it as a car border
		int trigger = 2;
		int consecutiveLines;
		bool terminateOuterLoop = false;
		cv::Vec4i carBorderLine;
		//the first carBorderLine from the bottom is the one returned
		while (it4!=liHorizontal.end() && !terminateOuterLoop) {
			it5 = it4 + 1;
			consecutiveLines = 0;
			while (it5!=liHorizontal.end()) {
				//distance between lines is less than
				if (std::abs((*it4)[1] - (*it5)[1]) < 70 && consecutiveLines < trigger){
					++consecutiveLines;
					if(consecutiveLines == trigger){
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

		//reload image to separately display the computing of all the parameters
		if(i < 10) image1 = image1 = cv::imread(std::string(video_folder)+"000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else if (i < 100) image1 = cv::imread(std::string(video_folder)+"00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
	  else image1 = cv::imread(std::string(video_folder)+"0000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);

    cv::Mat roiCarBumper(image1, cv::Rect(video_x0,video_y0,video_width,video_height));

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

		cv::rectangle(image1, cv::Rect(video_x0,video_y0,video_width,video_height), cv::Scalar(255),1);
		cv::line(roiCarBumper, cv::Point(carBorderLine[0],carBorderLine[1]),cv::Point(carBorderLine[2],carBorderLine[3]),cv::Scalar(255),3);

		cv::namedWindow("Front Car Bumper Detection");
		cv::imshow("Front Car Bumper Detection",image1);

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

    if(i < 10) image1 = cv::imread(std::string(video_folder)+"000000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
		else if (i < 100) image1 = cv::imread(std::string(video_folder)+"00000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);
	  else image1 = cv::imread(std::string(video_folder)+"0000000"+std::to_string(i)+".jpg",CV_LOAD_IMAGE_COLOR);

		cv::Mat roiFinalresult(image1, cv::Rect(video_x0,video_y0,video_width,video_height));

		// draw the resulting towards vector line
		cv::arrowedLine(roiFinalresult,
						 towardsVectorPoint1,
						 towardsVectorPoint2,
						 colorDangerous,
					   2,
					 	 8,
					   0,
					   0.5);

	  // draw the resulting away vector line
		cv::arrowedLine(roiFinalresult,
						 awayVectorPoint1,
						 awayVectorPoint2,
						 colorGood,
					   2,
						 8,
					   0,
					   0.5);

		cv::namedWindow("Collision detector");
		cv::imshow("Collision detector",image1);
		cv::waitKey(100);
	}

  cv::waitKey();
}
