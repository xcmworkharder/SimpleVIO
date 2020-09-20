CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(SIMPLE_VIO)

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g") # -O3最高代码级优化(o0最低) -g生成gdb调试信息 -wall

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # 自定义或统一存放cmake模块

FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(VTK REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem program_options system)
FIND_PACKAGE(Pangolin REQUIRED)
#FIND_PACKAGE(Sophus REQUIRED)

INCLUDE_DIRECTORIES(
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

ADD_LIBRARY(camera_model SHARED
        src/camodocal/chessboard/Chessboard.cc
        src/camodocal/calib/CameraCalibration.cc
        src/camodocal/camera_models/Camera.cc
        src/camodocal/camera_models/CameraFactory.cc
        src/camodocal/camera_models/CostFunctionFactory.cc
        src/camodocal/camera_models/PinholeCamera.cc
        src/camodocal/camera_models/CataCamera.cc
        src/camodocal/camera_models/EquidistantCamera.cc
        src/camodocal/camera_models/ScaramuzzaCamera.cc
        src/camodocal/sparse_graph/Transform.cc
        src/camodocal/gpl/gpl.cc
        src/camodocal/gpl/EigenQuaternionParameterization.cc
        )
TARGET_LINK_LIBRARIES(camera_model
        ${Boost_LIBRARIES} ${OPENCV_LIBS} ${CERES_LIBRARIES})

ADD_LIBRARY(MyVio SHARED
        src/system.cpp
        src/parameters.cpp
        src/estimator.cpp
        src/feature_tracker.cpp
        src/feature_manager.cpp

        src/initialization/initial_alignment.cpp
        src/initialization/initial_ex_rotation.cpp
        src/initialization/initial_sfm.cpp
        src/initialization/solve_5pts.cpp

        src/backend/vertex.cpp
        src/backend/edge.cpp
        src/backend/problem.cpp
        src/backend/vertex_pose.cpp
        src/backend/edge_reprojection.cpp
        src/backend/edge_imu.cpp
        src/backend/edge_prior.cpp
        src/backend/loss_function.cpp
        src/backend/imu_integration.cpp
        )
TARGET_LINK_LIBRARIES(MyVio
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES}
        camera_model)

ADD_EXECUTABLE(run_euroc app/run_euroc.cpp)
TARGET_LINK_LIBRARIES(run_euroc
        MyVio
        -lpthread
        -fopenmp
        )