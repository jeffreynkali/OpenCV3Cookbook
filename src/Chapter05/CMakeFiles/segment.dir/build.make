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
include Chapter05/CMakeFiles/segment.dir/depend.make

# Include the progress variables for this target.
include Chapter05/CMakeFiles/segment.dir/progress.make

# Include the compile flags for this target's objects.
include Chapter05/CMakeFiles/segment.dir/flags.make

Chapter05/CMakeFiles/segment.dir/segment.cpp.o: Chapter05/CMakeFiles/segment.dir/flags.make
Chapter05/CMakeFiles/segment.dir/segment.cpp.o: Chapter05/segment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Chapter05/CMakeFiles/segment.dir/segment.cpp.o"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/segment.dir/segment.cpp.o -c "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05/segment.cpp"

Chapter05/CMakeFiles/segment.dir/segment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/segment.dir/segment.cpp.i"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05/segment.cpp" > CMakeFiles/segment.dir/segment.cpp.i

Chapter05/CMakeFiles/segment.dir/segment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/segment.dir/segment.cpp.s"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05/segment.cpp" -o CMakeFiles/segment.dir/segment.cpp.s

Chapter05/CMakeFiles/segment.dir/segment.cpp.o.requires:

.PHONY : Chapter05/CMakeFiles/segment.dir/segment.cpp.o.requires

Chapter05/CMakeFiles/segment.dir/segment.cpp.o.provides: Chapter05/CMakeFiles/segment.dir/segment.cpp.o.requires
	$(MAKE) -f Chapter05/CMakeFiles/segment.dir/build.make Chapter05/CMakeFiles/segment.dir/segment.cpp.o.provides.build
.PHONY : Chapter05/CMakeFiles/segment.dir/segment.cpp.o.provides

Chapter05/CMakeFiles/segment.dir/segment.cpp.o.provides.build: Chapter05/CMakeFiles/segment.dir/segment.cpp.o


# Object files for target segment
segment_OBJECTS = \
"CMakeFiles/segment.dir/segment.cpp.o"

# External object files for target segment
segment_EXTERNAL_OBJECTS =

Chapter05/segment: Chapter05/CMakeFiles/segment.dir/segment.cpp.o
Chapter05/segment: Chapter05/CMakeFiles/segment.dir/build.make
Chapter05/segment: /usr/local/lib/libopencv_stitching.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_superres.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_videostab.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_aruco.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_bgsegm.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_bioinspired.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_ccalib.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_dpm.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_face.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_fuzzy.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_img_hash.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_line_descriptor.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_optflow.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_reg.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_rgbd.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_saliency.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_stereo.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_structured_light.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_surface_matching.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_tracking.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_xfeatures2d.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_ximgproc.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_xobjdetect.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_xphoto.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_shape.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_photo.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_datasets.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_plot.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_text.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_dnn.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_ml.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_video.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_calib3d.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_features2d.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_highgui.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_videoio.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_phase_unwrapping.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_flann.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_imgcodecs.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_objdetect.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_imgproc.3.4.0.dylib
Chapter05/segment: /usr/local/lib/libopencv_core.3.4.0.dylib
Chapter05/segment: Chapter05/CMakeFiles/segment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable segment"
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/segment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Chapter05/CMakeFiles/segment.dir/build: Chapter05/segment

.PHONY : Chapter05/CMakeFiles/segment.dir/build

Chapter05/CMakeFiles/segment.dir/requires: Chapter05/CMakeFiles/segment.dir/segment.cpp.o.requires

.PHONY : Chapter05/CMakeFiles/segment.dir/requires

Chapter05/CMakeFiles/segment.dir/clean:
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" && $(CMAKE_COMMAND) -P CMakeFiles/segment.dir/cmake_clean.cmake
.PHONY : Chapter05/CMakeFiles/segment.dir/clean

Chapter05/CMakeFiles/segment.dir/depend:
	cd "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05" "/Users/glenjeffreympoyonkali/Documents/University of Ottawa/Winter 2018/CSI 4533 - Méthodes informatisées en traitement d'images/Code/OpenCV3Cookbook/src/Chapter05/CMakeFiles/segment.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : Chapter05/CMakeFiles/segment.dir/depend

