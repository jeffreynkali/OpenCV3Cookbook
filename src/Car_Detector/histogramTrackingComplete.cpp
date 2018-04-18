#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>

#include "histogram.h"

int main(){

  // read the first image as a grayscale image
	cv::Mat image0 =  cv::imread("images/0000000000.jpg", 0);

  // define image ROI around car black suv
  //int xo=411, yo=170;
	//int width=100, height=71;

	// define image ROI around car gray full size
  int xo=160, yo=190;
	int width=115, height=70;

	// draw a rectangle around target object
	cv::rectangle(image0,cv::Rect(xo,yo,width,height),0);
	cv::namedWindow("Initial Image");
	cv::imshow("Initial Image",image0);

	cv::Mat image0ROI(image0, cv::Rect(xo,yo,width,height));// ROI size

  // histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	// compute histogram over image roi
	cv::Mat refHistogram= h.getHistogram(image0ROI);

  cv::namedWindow("Reference Histogram");
	cv::imshow("Reference Histogram",h.getHistogramImage(image0ROI,16));


	double maxSimilarity=0.0;
	int xbest, ybest;
  // search in all 81 images one at a time
	for (int i = 1; i < 82; i++){
    cv::Mat imageNext;
		if(i<10) imageNext = cv::imread("images/000000000"+std::to_string(i)+".jpg",0);
		else imageNext = cv::imread("images/00000000"+std::to_string(i)+".jpg",0);
		if (!imageNext.data) return 0;

		cv::Mat imageNextROI(imageNext, cv::Rect(xo,yo,width,height));
	  // compute histogram over image roi
		cv::Mat histogram= h.getHistogram(imageNextROI);

	  cv::namedWindow("Image Next Histogram");
		cv::imshow("Image Next Histogram",h.getHistogramImage(imageNextROI,16));

		// The car may have moved upwards but it must be ahead of the actual car
	  // that is taking the pictures so we can assume that the height is only
	  //gonna decrease towards height
	  // for the horizontal line we are going to sweep the whole picture width
		for (int y=yo-20; y <= yo + height + 20; y++) {
		//for (int y=0; y < imageNext.rows - height; y++) {
			for (int x=0; x < imageNext.cols - width; x++) {

	      //compute histogram of actual possibility
	      cv::Mat roi(imageNext, cv::Rect(x,y,width,height));
	      // compute histogram over image roi
	    	histogram= h.getHistogram(roi);
				// compute distance with reference histogram
				double distance= cv::compareHist(refHistogram,histogram, cv::HISTCMP_INTERSECT);
				// find position of most similar histogram
				if (distance>maxSimilarity) {
					xbest= x;
					ybest= y;
					maxSimilarity= distance;
				}
			}
		}

		std::cout << "Best solution= (" << xbest << "," << ybest << ")=" << maxSimilarity << std::endl;

	  // draw rectangle at best location
	  cv::rectangle(imageNext,cv::Rect(xbest,ybest,width,height),0,5);
	  // draw rectangle around search area
	  cv::rectangle(imageNext,cv::Rect(0,yo-20,imageNext.cols,height+20),255);

	  cv::imshow("Object location ",imageNext);
		cv::waitKey(10);

		maxSimilarity=0.0;
		xo = xbest;
		yo = ybest;
	}

  cv::waitKey();
}
