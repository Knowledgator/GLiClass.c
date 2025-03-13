cmake_minimum_required(VERSION 3.18)
project(gliclass_example LANGUAGES C VERSION 0.0.1)

if (MSVC)
    add_compile_options(/W4)
else()
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(GLICLASS_ROOTDIR ../)

add_subdirectory(${GLICLASS_ROOTDIR} GLiClass)

add_executable(inference_multi_label_batch inference_multi_label_batch.c)

target_include_directories(inference_multi_label_batch PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_multi_label_batch GLiClass)


add_executable(inference_single_label_batch inference_single_label_batch.c)

target_include_directories(inference_single_label_batch PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_single_label_batch GLiClass)


add_executable(inference_multi_label_single inference_multi_label_single.c)

target_include_directories(inference_multi_label_single PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_multi_label_single GLiClass)


add_executable(inference_single_label_single inference_single_label_single.c)

target_include_directories(inference_single_label_single PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_single_label_single GLiClass)