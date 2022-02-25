project(task)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

set(SOURCE 
    main.cpp 
    perceptron.cpp 
    perceptron2.cpp 
    sigmoid.cpp 
    teacher.cpp
    painter.cpp)

add_executable(${PROJECT_NAME} ${SOURCE})
target_link_libraries(${PROJECT_NAME} pthread boost_serialization boost_system)
