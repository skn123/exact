# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.13.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.13.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/a.e./Dropbox/exact

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/a.e./Dropbox/exact/build

# Include any dependencies generated for this target.
include mpi/CMakeFiles/exact_mpi.dir/depend.make

# Include the progress variables for this target.
include mpi/CMakeFiles/exact_mpi.dir/progress.make

# Include the compile flags for this target's objects.
include mpi/CMakeFiles/exact_mpi.dir/flags.make

mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o: mpi/CMakeFiles/exact_mpi.dir/flags.make
mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o: ../mpi/exact_mpi.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/a.e./Dropbox/exact/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o"
	cd /Users/a.e./Dropbox/exact/build/mpi && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o -c /Users/a.e./Dropbox/exact/mpi/exact_mpi.cxx

mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exact_mpi.dir/exact_mpi.cxx.i"
	cd /Users/a.e./Dropbox/exact/build/mpi && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/a.e./Dropbox/exact/mpi/exact_mpi.cxx > CMakeFiles/exact_mpi.dir/exact_mpi.cxx.i

mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exact_mpi.dir/exact_mpi.cxx.s"
	cd /Users/a.e./Dropbox/exact/build/mpi && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/a.e./Dropbox/exact/mpi/exact_mpi.cxx -o CMakeFiles/exact_mpi.dir/exact_mpi.cxx.s

# Object files for target exact_mpi
exact_mpi_OBJECTS = \
"CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o"

# External object files for target exact_mpi
exact_mpi_EXTERNAL_OBJECTS =

mpi/exact_mpi: mpi/CMakeFiles/exact_mpi.dir/exact_mpi.cxx.o
mpi/exact_mpi: mpi/CMakeFiles/exact_mpi.dir/build.make
mpi/exact_mpi: cnn/libexact_strategy.a
mpi/exact_mpi: image_tools/libexact_image_tools.a
mpi/exact_mpi: common/libexact_common.a
mpi/exact_mpi: /Users/a.e./anaconda2/lib/libmpi_cxx.dylib
mpi/exact_mpi: /Users/a.e./anaconda2/lib/libmpi.dylib
mpi/exact_mpi: /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk/usr/lib/libm.tbd
mpi/exact_mpi: /usr/local/lib/libmysqlclient.dylib
mpi/exact_mpi: /usr/local/lib/libtiff.dylib
mpi/exact_mpi: mpi/CMakeFiles/exact_mpi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/a.e./Dropbox/exact/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable exact_mpi"
	cd /Users/a.e./Dropbox/exact/build/mpi && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exact_mpi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mpi/CMakeFiles/exact_mpi.dir/build: mpi/exact_mpi

.PHONY : mpi/CMakeFiles/exact_mpi.dir/build

mpi/CMakeFiles/exact_mpi.dir/clean:
	cd /Users/a.e./Dropbox/exact/build/mpi && $(CMAKE_COMMAND) -P CMakeFiles/exact_mpi.dir/cmake_clean.cmake
.PHONY : mpi/CMakeFiles/exact_mpi.dir/clean

mpi/CMakeFiles/exact_mpi.dir/depend:
	cd /Users/a.e./Dropbox/exact/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/a.e./Dropbox/exact /Users/a.e./Dropbox/exact/mpi /Users/a.e./Dropbox/exact/build /Users/a.e./Dropbox/exact/build/mpi /Users/a.e./Dropbox/exact/build/mpi/CMakeFiles/exact_mpi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mpi/CMakeFiles/exact_mpi.dir/depend
