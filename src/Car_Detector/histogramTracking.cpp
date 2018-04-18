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
  int xo=411, yo=170;
	int width=100, height=71;
	cv::Mat image0ROI(image0, cv::Rect(xo,yo,width,height));// ROI size

  // histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	// compute histogram over image roi
	cv::Mat refHistogram= h.getHistogram(image0ROI);

  cv::namedWindow("Reference Histogram");
	cv::imshow("Reference Histogram",h.getHistogramImage(image0ROI,16));

  // search in second image
	cv::Mat image1 = cv::imread("images/0000000001.jpg",0);
	if (!image1.data) return 0;

  cv::Mat image1ROI(image1, cv::Rect(xo,yo,width,height));
  // compute histogram over image roi
	cv::Mat histogram= h.getHistogram(image1ROI);

  cv::namedWindow("Image 1 Histogram");
	cv::imshow("Image 1 Histogram",h.getHistogramImage(image1ROI,16));

  double maxSimilarity=0.0;
	int xbest, ybest;
	// The car may have moved upwards but it must be ahead of the actual car
  // that is taking the pictures so we can assume that the height is only
  //gonna decrease towards height
  // for the horizontal line we are going to sweep the whole picture width
	for (int y=yo+height; y>=height; y--) {
		for (int x=0; x<image1.cols-width; x++) {

      //compute histogram of actual possibility
      cv::Mat roi(image1, cv::Rect(x,y,width,height));
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

  // draw a rectangle around target object
  cv::rectangle(image0,cv::Rect(xo,yo,width,height),0);
  cv::namedWindow("Initial Image");
  cv::imshow("Initial Image",image0);

  // draw rectangle at best location
  cv::rectangle(image1,cv::Rect(xbest,ybest,width,height),0);
  // draw rectangle around search area
  cv::rectangle(image1,cv::Rect(0,0,image1.cols,yo+height),255);
  cv::namedWindow("Object location");
  cv::imshow("Object location",image1);

  cv::waitKey();
}
