# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build

# Include any dependencies generated for this target.
include CMakeFiles/lenet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lenet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lenet.dir/flags.make

CMakeFiles/lenet.dir/src/main.cpp.o: CMakeFiles/lenet.dir/flags.make
CMakeFiles/lenet.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lenet.dir/src/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lenet.dir/src/main.cpp.o -c /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/main.cpp

CMakeFiles/lenet.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lenet.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/main.cpp > CMakeFiles/lenet.dir/src/main.cpp.i

CMakeFiles/lenet.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lenet.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/main.cpp -o CMakeFiles/lenet.dir/src/main.cpp.s

CMakeFiles/lenet.dir/src/tools.cpp.o: CMakeFiles/lenet.dir/flags.make
CMakeFiles/lenet.dir/src/tools.cpp.o: ../src/tools.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lenet.dir/src/tools.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lenet.dir/src/tools.cpp.o -c /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/tools.cpp

CMakeFiles/lenet.dir/src/tools.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lenet.dir/src/tools.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/tools.cpp > CMakeFiles/lenet.dir/src/tools.cpp.i

CMakeFiles/lenet.dir/src/tools.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lenet.dir/src/tools.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/src/tools.cpp -o CMakeFiles/lenet.dir/src/tools.cpp.s

# Object files for target lenet
lenet_OBJECTS = \
"CMakeFiles/lenet.dir/src/main.cpp.o" \
"CMakeFiles/lenet.dir/src/tools.cpp.o"

# External object files for target lenet
lenet_EXTERNAL_OBJECTS =

../bin/lenet: CMakeFiles/lenet.dir/src/main.cpp.o
../bin/lenet: CMakeFiles/lenet.dir/src/tools.cpp.o
../bin/lenet: CMakeFiles/lenet.dir/build.make
../bin/lenet: /home/saber/Desktop/GitHub/pytorch/torch/lib/libtorch.so
../bin/lenet: /home/saber/Desktop/GitHub/pytorch/torch/lib/libc10.so
../bin/lenet: /usr/lib/x86_64-linux-gnu/libpython3.8.so
../bin/lenet: /home/saber/Desktop/GitHub/pytorch/torch/lib/libc10.so
../bin/lenet: CMakeFiles/lenet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/lenet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lenet.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/cmake -E make_directory output

# Rule to build all files generated by this target.
CMakeFiles/lenet.dir/build: ../bin/lenet

.PHONY : CMakeFiles/lenet.dir/build

CMakeFiles/lenet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lenet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lenet.dir/clean

CMakeFiles/lenet.dir/depend:
	cd /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build /home/saber/Desktop/GitHub/Pytorch_CPP/d2l_pytorch_cpp/Ch07_Deep_Learning_Computation/LeNET/build/CMakeFiles/lenet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lenet.dir/depend
