cmake_minimum_required(VERSION 3.18)
project(gliclass_example LANGUAGES C VERSION 0.0.1)

if (MSVC)
    add_compile_options(/W4)
else()
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)
endif()

set(GLICLASS_ROOTDIR ../)

set(BUILD_TARGET "CPU" CACHE STRING "Choose the build target: CPU or GPU")

add_subdirectory(${GLICLASS_ROOTDIR} GLiClass)

if (ONNX)
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
endif()

if (ONNX_OPENVINO)
    add_executable(inference_ort_openvino inference_ort_openvino.c)
    
    if (${OPENVINO_PATH} STREQUAL "")
        find_library(OPENVINO_LIB
        NAMES openvino_c libopenvino_c
        )
    else()    
        find_library(OPENVINO_LIB
        NAMES openvino_c libopenvino_c
        PATHS ${OPENVINO_LIB_PATH}
        NO_DEFAULT_PATH
        )
    endif()
    
    if(NOT OPENVINO_LIB)
        message(FATAL_ERROR "OpenVino lib is not found at ${OPENVINO_LIB_PATH}")
    endif()

    target_include_directories(inference_ort_openvino PRIVATE 
        ${GLICLASS_ROOTDIR}/include
        ${OPENVINO_PATH}/include)
    target_link_libraries(inference_ort_openvino 
        GLiClass 
        ${OPENVINO_LIB})
endif()

if(OPENVINO)
    add_executable(inference_openvino_runtime inference_openvino_runtime.c)
    target_include_directories(inference_openvino_runtime PRIVATE ${GLICLASS_ROOTDIR}/include)
    target_link_libraries(inference_openvino_runtime PRIVATE GLiClass)
endif()

if(ONNX_CUDA)
    add_executable(inference_ort_cuda inference_ort_cuda.c)
    target_include_directories(inference_ort_cuda PRIVATE ${GLICLASS_ROOTDIR}/include)
    target_link_libraries(inference_ort_cuda GLiClass)
endif()