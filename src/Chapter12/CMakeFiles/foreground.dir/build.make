# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.10.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.10.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src"

# Include any dependencies generated for this target.
include Chapter12/CMakeFiles/foreground.dir/depend.make

# Include the progress variables for this target.
include Chapter12/CMakeFiles/foreground.dir/progress.make

# Include the compile flags for this target's objects.
include Chapter12/CMakeFiles/foreground.dir/flags.make

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o: Chapter12/CMakeFiles/foreground.dir/flags.make
Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o: Chapter12/foreground.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/foreground.dir/foreground.cpp.o -c "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12/foreground.cpp"

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/foreground.dir/foreground.cpp.i"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12/foreground.cpp" > CMakeFiles/foreground.dir/foreground.cpp.i

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/foreground.dir/foreground.cpp.s"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12/foreground.cpp" -o CMakeFiles/foreground.dir/foreground.cpp.s

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.requires:

.PHONY : Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.requires

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.provides: Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.requires
	$(MAKE) -f Chapter12/CMakeFiles/foreground.dir/build.make Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.provides.build
.PHONY : Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.provides

Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.provides.build: Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o


# Object files for target foreground
foreground_OBJECTS = \
"CMakeFiles/foreground.dir/foreground.cpp.o"

# External object files for target foreground
foreground_EXTERNAL_OBJECTS =

Chapter12/foreground: Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o
Chapter12/foreground: Chapter12/CMakeFiles/foreground.dir/build.make
Chapter12/foreground: /usr/local/lib/libopencv_stitching.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_superres.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_videostab.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_aruco.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_bgsegm.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_bioinspired.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_ccalib.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_dpm.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_face.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_fuzzy.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_img_hash.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_line_descriptor.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_optflow.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_reg.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_rgbd.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_saliency.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_stereo.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_structured_light.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_surface_matching.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_tracking.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_xfeatures2d.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_ximgproc.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_xobjdetect.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_xphoto.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_shape.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_photo.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_datasets.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_plot.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_text.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_dnn.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_ml.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_video.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_calib3d.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_features2d.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_highgui.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_videoio.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_phase_unwrapping.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_flann.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_imgcodecs.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_objdetect.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_imgproc.3.4.0.dylib
Chapter12/foreground: /usr/local/lib/libopencv_core.3.4.0.dylib
Chapter12/foreground: Chapter12/CMakeFiles/foreground.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable foreground"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/foreground.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Chapter12/CMakeFiles/foreground.dir/build: Chapter12/foreground

.PHONY : Chapter12/CMakeFiles/foreground.dir/build

Chapter12/CMakeFiles/foreground.dir/requires: Chapter12/CMakeFiles/foreground.dir/foreground.cpp.o.requires

.PHONY : Chapter12/CMakeFiles/foreground.dir/requires

Chapter12/CMakeFiles/foreground.dir/clean:
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" && $(CMAKE_COMMAND) -P CMakeFiles/foreground.dir/cmake_clean.cmake
.PHONY : Chapter12/CMakeFiles/foreground.dir/clean

Chapter12/CMakeFiles/foreground.dir/depend:
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter12/CMakeFiles/foreground.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : Chapter12/CMakeFiles/foreground.dir/depend

