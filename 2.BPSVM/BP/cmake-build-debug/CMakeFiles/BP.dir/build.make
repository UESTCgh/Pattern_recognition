# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "D:\software\JetBrains\CLion 2024.2.2\bin\cmake\win\x64\bin\cmake.exe"

# The command to remove a file.
RM = "D:\software\JetBrains\CLion 2024.2.2\bin\cmake\win\x64\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\GitHub\AI_VI\2.BPSVM\BP

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/BP.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/BP.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/BP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BP.dir/flags.make

CMakeFiles/BP.dir/main.cpp.obj: CMakeFiles/BP.dir/flags.make
CMakeFiles/BP.dir/main.cpp.obj: E:/GitHub/AI_VI/2.BPSVM/BP/main.cpp
CMakeFiles/BP.dir/main.cpp.obj: CMakeFiles/BP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/BP.dir/main.cpp.obj"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/BP.dir/main.cpp.obj -MF CMakeFiles\BP.dir\main.cpp.obj.d -o CMakeFiles\BP.dir\main.cpp.obj -c E:\GitHub\AI_VI\2.BPSVM\BP\main.cpp

CMakeFiles/BP.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/BP.dir/main.cpp.i"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\GitHub\AI_VI\2.BPSVM\BP\main.cpp > CMakeFiles\BP.dir\main.cpp.i

CMakeFiles/BP.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/BP.dir/main.cpp.s"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\GitHub\AI_VI\2.BPSVM\BP\main.cpp -o CMakeFiles\BP.dir\main.cpp.s

CMakeFiles/BP.dir/bp.cpp.obj: CMakeFiles/BP.dir/flags.make
CMakeFiles/BP.dir/bp.cpp.obj: E:/GitHub/AI_VI/2.BPSVM/BP/bp.cpp
CMakeFiles/BP.dir/bp.cpp.obj: CMakeFiles/BP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/BP.dir/bp.cpp.obj"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/BP.dir/bp.cpp.obj -MF CMakeFiles\BP.dir\bp.cpp.obj.d -o CMakeFiles\BP.dir\bp.cpp.obj -c E:\GitHub\AI_VI\2.BPSVM\BP\bp.cpp

CMakeFiles/BP.dir/bp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/BP.dir/bp.cpp.i"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\GitHub\AI_VI\2.BPSVM\BP\bp.cpp > CMakeFiles\BP.dir\bp.cpp.i

CMakeFiles/BP.dir/bp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/BP.dir/bp.cpp.s"
	D:\software\JETBRA~1\CLION2~1.2\bin\mingw\bin\G__~1.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\GitHub\AI_VI\2.BPSVM\BP\bp.cpp -o CMakeFiles\BP.dir\bp.cpp.s

# Object files for target BP
BP_OBJECTS = \
"CMakeFiles/BP.dir/main.cpp.obj" \
"CMakeFiles/BP.dir/bp.cpp.obj"

# External object files for target BP
BP_EXTERNAL_OBJECTS =

BP.exe: CMakeFiles/BP.dir/main.cpp.obj
BP.exe: CMakeFiles/BP.dir/bp.cpp.obj
BP.exe: CMakeFiles/BP.dir/build.make
BP.exe: CMakeFiles/BP.dir/linkLibs.rsp
BP.exe: CMakeFiles/BP.dir/objects1.rsp
BP.exe: CMakeFiles/BP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable BP.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\BP.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BP.dir/build: BP.exe
.PHONY : CMakeFiles/BP.dir/build

CMakeFiles/BP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\BP.dir\cmake_clean.cmake
.PHONY : CMakeFiles/BP.dir/clean

CMakeFiles/BP.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\GitHub\AI_VI\2.BPSVM\BP E:\GitHub\AI_VI\2.BPSVM\BP E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug E:\GitHub\AI_VI\2.BPSVM\BP\cmake-build-debug\CMakeFiles\BP.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/BP.dir/depend

