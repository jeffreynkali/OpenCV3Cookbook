# cmake for Car Detector based on histograms
# your opencv/build directory should be in your system PATH

# set minimum required version for cmake
cmake_minimum_required(VERSION 2.8)

# add executable
add_executable( sobelTracking sobelTracking.cpp)
add_executable( carTracking carTracking.cpp)

# link libraries
target_link_libraries( sobelTracking ${OpenCV_LIBS})
target_link_libraries( carTracking ${OpenCV_LIBS})
