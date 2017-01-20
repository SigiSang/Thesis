cmake_minimum_required( VERSION 3.1 )
project( main )

set( dir_src src )
set( dir_src_md ${dir_src}/motion_detection )
#set( dir_src_zloko ${dir_src}/Zlokolica )

file( GLOB_RECURSE headers ${dir_src}/*.h )
file( GLOB sources ${dir_src}/main.cpp )
#file( GLOB sources ${dir_src}/main.cpp ${dir_src_zloko}/*.cc ${dir_src_zloko}/*.cpp )

include_directories( ${dir_src} )
include_directories( ${dir_src}/headers )
include_directories( ${dir_src_md} )
#include_directories( ${dir_src_zloko} )

find_package( OpenCV REQUIRED )
add_executable( main ${headers} ${sources} )
target_link_libraries( main ${OpenCV_LIBS} )
target_compile_options(main PUBLIC -std=c++11)
