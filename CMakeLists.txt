cmake_minimum_required(VERSION 3.20)
project(FastCAT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_UI "Enabled UI" ON)
option(USE_MIXED_PRECSION "Enabled Mixed Precision" ON)
option(TIME_PROFILE "Enabled Time Profiling" OFF)
option(DEBUG "Enabled Debug Mode" OFF)

include_directories(${PROJECT_SOURCE_DIR})

if(USE_UI)
    include(FetchContent)
    FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/ftxui
    GIT_TAG v6.1.9
    )
    FetchContent_MakeAvailable(ftxui)
endif()

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")
add_library(fastcat_lib ${SOURCES})
target_include_directories(fastcat_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)


if(USE_UI)
    target_link_libraries(fastcat_lib PUBLIC 
        Eigen3::Eigen 
        OpenMP::OpenMP_CXX 
        Boost::boost 
        ftxui::component 
        ftxui::dom 
        ftxui::screen
    )

    add_executable(main src/main.cpp)
    target_link_libraries(main PRIVATE 
        fastcat_lib 
        Eigen3::Eigen 
        OpenMP::OpenMP_CXX 
        Boost::boost 
        ftxui::component 
        ftxui::dom 
        ftxui::screen
    )
    add_compile_definitions(USE_UI=1)
else()
    target_link_libraries(fastcat_lib PUBLIC 
        Eigen3::Eigen 
        OpenMP::OpenMP_CXX 
        Boost::boost 
    )

    add_executable(main src/main.cpp)
    target_link_libraries(main PRIVATE 
        fastcat_lib 
        Eigen3::Eigen 
        OpenMP::OpenMP_CXX 
        Boost::boost 
    )
endif()

target_compile_options(fastcat_lib PRIVATE -march=native -O3)
target_compile_options(main PRIVATE -march=native -O3)

if(DEBUG)
    target_compile_options(fastcat_lib PRIVATE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(fastcat_lib PRIVATE -fsanitize=address)
    target_compile_options(main PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
    target_link_options(main PRIVATE -fsanitize=address)
endif()

if(USE_MIXED_PRECISION)
    add_compile_definitions(MIXED_PRECISION=1)
endif()

if(TIME_PROFILE)
    add_compile_definitions(TIME_PROFILE=1)
endif()