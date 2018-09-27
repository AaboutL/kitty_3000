cmake_minimum_required(VERSION 3.5.1)
project(align_3000fps_cv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -std=c++14 -march=native -fopenmp")

find_package(OpenCV HINTS "/home/public/nfs72/3000_train/opencv-3.4/install/share/OpenCV")
#find_package(Boost 1.54.0 COMPONENTS filesystem system program_options)
#add_subdirectory(/home/slam/workspace/DL/dlib/ dlib_build)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
#        ${Boost_INCLUDE_DIR}
)
add_library(${PROJECT_NAME} SHARED
        src/facemark_train.cpp
        src/facemarkLBF.cpp
#        src/FaceDetection.cpp
        )
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
#        dlib::dlib
        )

add_executable(train_ibugs test/train_ibugs.cc)
target_link_libraries(train_ibugs
        ${PROJECT_NAME}
#        dlib::dlib
        )

add_executable(train_300W_LP test/train_300W_LP.cc)
target_link_libraries(train_300W_LP
        ${PROJECT_NAME}
        )

add_executable(train_cv test/train_cv.cc)
target_link_libraries(train_cv
        ${PROJECT_NAME}
        #        dlib::dlib
        )
add_executable(train_offline test/train_offline.cc)
target_link_libraries(train_offline
        ${PROJECT_NAME}
        )
add_executable(train_offline_13 test/train_offline_13.cc)
target_link_libraries(train_offline_13
        ${PROJECT_NAME}
        )
add_executable(train_offline_32 test/train_offline_32.cc)
target_link_libraries(train_offline_32
        ${PROJECT_NAME}
        )
add_executable(train_no_meimao test/train_no_meimao.cc)
target_link_libraries(train_no_meimao
        ${PROJECT_NAME}
        )

add_executable(test_model test/test_model.cc)
target_link_libraries(test_model
        ${PROJECT_NAME}
#        dlib::dlib
        )

add_executable(test_model_offline test/test_model_offline.cc)
target_link_libraries(test_model_offline
        ${PROJECT_NAME}
        #        dlib::dlib
        )

add_executable(test_model_cv test/test_model_cv.cc)
target_link_libraries(test_model_cv
        ${PROJECT_NAME}
        #        dlib::dlib
        )
add_executable(test_offline test/test_offline.cc)
target_link_libraries(test_offline
        ${PROJECT_NAME}
        #        dlib::dlib
        )
add_definitions(-D_OPENMP)
