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

set(BUILD_TARGET "CPU" CACHE STRING "Choose the build target: CPU or GPU")

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

add_executable(inference_multi_label_single_repeat inference_multi_label_single_repeat.c)
target_include_directories(inference_multi_label_single_repeat PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_multi_label_single_repeat GLiClass)

add_executable(inference_multi_label_single_config inference_multi_label_single_config.c)
target_include_directories(inference_multi_label_single_config PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_multi_label_single_config GLiClass)

add_executable(inference_single_label_single inference_single_label_single.c)
target_include_directories(inference_single_label_single PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_single_label_single GLiClass)

add_executable(inference_custom_ort inference_custom_ort.c)
target_include_directories(inference_custom_ort PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_custom_ort GLiClass)

add_executable(inference_openvino inference_openvino.c)
if (WIN32)
    target_include_directories(inference_openvino PRIVATE "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/include")
    target_link_directories(inference_openvino PRIVATE "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/lib/intel64/Release")
target_link_libraries(inference_openvino "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime/lib/intel64/Release/openvino_c.lib")
else()
    target_include_directories(inference_openvino PUBLIC "/opt/intel/openvino_2024/runtime/include")
    target_link_directories(inference_openvino PUBLIC "/opt/intel/openvino_2024/runtime/lib/intel64")
    target_link_libraries(inference_openvino "/opt/intel/openvino_2024/runtime/lib/intel64/libopenvino_c.so")
endif()
target_include_directories(inference_openvino PRIVATE ${GLICLASS_ROOTDIR}/include)
target_link_libraries(inference_openvino GLiClass)

if(BUILD_TARGET STREQUAL "GPU")
    add_executable(inference_cuda inference_cuda.c)
    target_include_directories(inference_cuda PRIVATE ${GLICLASS_ROOTDIR}/include)
    target_link_libraries(inference_cuda GLiClass)
endif()