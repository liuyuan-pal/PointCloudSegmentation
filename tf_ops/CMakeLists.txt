cmake_minimum_required(VERSION 3.5)
project(tf_ops)

#add_subdirectory(PoolingIndex)
link_directories(/home/pal/lib/lib)
include_directories(/home/pal/lib/include)
find_package (Threads)

add_executable(test_log4cpp test_log4cpp.cpp)
target_link_libraries(test_log4cpp log4cpp ${CMAKE_THREAD_LIBS_INIT})