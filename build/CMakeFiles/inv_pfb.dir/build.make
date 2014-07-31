# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/vusi/ska-ddc-read-only/inv_pfb

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vusi/ska-ddc-read-only/inv_pfb/build

# Include any dependencies generated for this target.
include CMakeFiles/inv_pfb.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/inv_pfb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/inv_pfb.dir/flags.make

CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o: CMakeFiles/inv_pfb.dir/inv_pfb_generated_inv_pfb.cu.o.depend
CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o: CMakeFiles/inv_pfb.dir/inv_pfb_generated_inv_pfb.cu.o.cmake
CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o: ../inv_pfb.cu
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Building NVCC (Device) object CMakeFiles/inv_pfb.dir//./inv_pfb_generated_inv_pfb.cu.o"
	cd /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir && /usr/bin/cmake -E make_directory /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir//.
	cd /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir//./inv_pfb_generated_inv_pfb.cu.o -D generated_cubin_file:STRING=/home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir//./inv_pfb_generated_inv_pfb.cu.o.cubin.txt -P /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir//inv_pfb_generated_inv_pfb.cu.o.cmake

CMakeFiles/inv_pfb.dir/driver.cpp.o: CMakeFiles/inv_pfb.dir/flags.make
CMakeFiles/inv_pfb.dir/driver.cpp.o: ../driver.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/inv_pfb.dir/driver.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/inv_pfb.dir/driver.cpp.o -c /home/vusi/ska-ddc-read-only/inv_pfb/driver.cpp

CMakeFiles/inv_pfb.dir/driver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/inv_pfb.dir/driver.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vusi/ska-ddc-read-only/inv_pfb/driver.cpp > CMakeFiles/inv_pfb.dir/driver.cpp.i

CMakeFiles/inv_pfb.dir/driver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/inv_pfb.dir/driver.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vusi/ska-ddc-read-only/inv_pfb/driver.cpp -o CMakeFiles/inv_pfb.dir/driver.cpp.s

CMakeFiles/inv_pfb.dir/driver.cpp.o.requires:
.PHONY : CMakeFiles/inv_pfb.dir/driver.cpp.o.requires

CMakeFiles/inv_pfb.dir/driver.cpp.o.provides: CMakeFiles/inv_pfb.dir/driver.cpp.o.requires
	$(MAKE) -f CMakeFiles/inv_pfb.dir/build.make CMakeFiles/inv_pfb.dir/driver.cpp.o.provides.build
.PHONY : CMakeFiles/inv_pfb.dir/driver.cpp.o.provides

CMakeFiles/inv_pfb.dir/driver.cpp.o.provides.build: CMakeFiles/inv_pfb.dir/driver.cpp.o

# Object files for target inv_pfb
inv_pfb_OBJECTS = \
"CMakeFiles/inv_pfb.dir/driver.cpp.o"

# External object files for target inv_pfb
inv_pfb_EXTERNAL_OBJECTS = \
"/home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o"

inv_pfb: CMakeFiles/inv_pfb.dir/driver.cpp.o
inv_pfb: CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o
inv_pfb: CMakeFiles/inv_pfb.dir/build.make
inv_pfb: /home/chris/CUDA/lib64/libcudart.so
inv_pfb: /home/chris/CUDA/lib64/libcufft.so
inv_pfb: CMakeFiles/inv_pfb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable inv_pfb"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/inv_pfb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/inv_pfb.dir/build: inv_pfb
.PHONY : CMakeFiles/inv_pfb.dir/build

CMakeFiles/inv_pfb.dir/requires: CMakeFiles/inv_pfb.dir/driver.cpp.o.requires
.PHONY : CMakeFiles/inv_pfb.dir/requires

CMakeFiles/inv_pfb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/inv_pfb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/inv_pfb.dir/clean

CMakeFiles/inv_pfb.dir/depend: CMakeFiles/inv_pfb.dir/./inv_pfb_generated_inv_pfb.cu.o
	cd /home/vusi/ska-ddc-read-only/inv_pfb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vusi/ska-ddc-read-only/inv_pfb /home/vusi/ska-ddc-read-only/inv_pfb /home/vusi/ska-ddc-read-only/inv_pfb/build /home/vusi/ska-ddc-read-only/inv_pfb/build /home/vusi/ska-ddc-read-only/inv_pfb/build/CMakeFiles/inv_pfb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/inv_pfb.dir/depend

