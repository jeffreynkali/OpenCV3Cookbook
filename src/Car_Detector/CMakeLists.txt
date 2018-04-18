# cmake for Car Detector based on histograms
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( histogramTracking histogramTracking.cpp)
add_executable( histogramTrackingComplete histogramTrackingComplete.cpp)

# link libraries
target_link_libraries( histogramTracking ${OpenCV_LIBS})
target_link_libraries( histogramTrackingComplete ${OpenCV_LIBS})
