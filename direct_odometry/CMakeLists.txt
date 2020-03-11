cmake_minimum_required(VERSION 2.8)
project(playground CXX)

find_package(OpenCV 3.4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_compile_features(${PROJECT_NAME}
PUBLIC
    cxx_std_11
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBRARIES}
)