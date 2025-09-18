cmake_minimum_required(VERSION 3.20)
project(FastCAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${PROJECT_SOURCE_DIR})
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")
add_library(fastcat_lib ${SOURCES})
target_include_directories(fastcat_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(fastcat_lib PUBLIC Eigen3::Eigen)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE fastcat_lib)

target_compile_options(fastcat_lib PRIVATE -march=native -O3)
target_compile_options(main PRIVATE -march=native -O3)
#target_compile_options(fastcat_lib PRIVATE -fsanitize=address -fno-omit-frame-pointer)
#target_link_options(fastcat_lib PRIVATE -fsanitize=address)
#target_compile_options(main PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
#target_link_options(main PRIVATE -fsanitize=address)
#add_compile_definitions(TEST_RUN=1)