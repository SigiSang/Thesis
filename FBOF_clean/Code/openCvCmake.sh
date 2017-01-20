#!/bin/bash
if [ $# != 1 ]; then
	echo "Use: openCvCmake.bash [projectName]">&2
	exit 1
fi

cd $PWD

projectName=$1
dir_src="src"
dir_hds="${dir_src}/headers"

for i in $dir_src $dir_hds "CMake_$projectName"; do
	if [ ! -d $i ]; then
		mkdir $i 1>&2 2>/dev/null;
	fi
done;

touch $projectName

if [ ! -f CMakeLists.txt ]; then
	echo "ERROR: No CMakeLists.txt file found, please configure one!" 1>&2;
fi;

cd "CMake_$projectName"
cmake ..
make
cp $projectName ../bin/