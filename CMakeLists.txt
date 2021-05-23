cmake_minimum_required(VERSION 3.0)

project(chapter1)
include_directories("/usr/local/include")
find_package( OpenCV REQUIRED )

add_executable(chapter1 augmentedObject.cc glad.c)

install(TARGETS chapter1 RUNTIME DESTINATION bin)

target_link_libraries(chapter1 ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

target_link_libraries(chapter1
    glfw
    glut
    GL
    dl
)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(chapter1 glfw)
# 
# find_package(glut REQUIRED)
# target_link_libraries(chapter1 glut)

include_directories(${SOURCE_FILES})
