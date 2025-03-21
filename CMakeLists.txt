cmake_minimum_required(VERSION 3.18)
project(GLiClass C CXX)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Set C standard
if (MSVC)
else()
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcommon")
endif()


# Include the headers
include_directories(include)

# Add option to choose between CPU and GPU
set(BUILD_TARGET "CPU" CACHE STRING "Choose the build target: CPU or GPU")
set_property(CACHE BUILD_TARGET PROPERTY STRINGS "CPU" "GPU")

message(STATUS "Build target set to: ${BUILD_TARGET}")

############################################ ONNX ###########################################

# Path to ONNXRuntime 
set(ONNXRUNTIME_PATH "./onnxruntime-linux-x64-1.19.2" CACHE STRING "Onnxruntime path")

if(BUILD_TARGET STREQUAL "GPU")
  # Perform checks for GPU, NVIDIA and cuDNN availability
  execute_process(
    COMMAND nvidia-smi
    RESULT_VARIABLE NVIDIA_SMI_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )

  execute_process(
    COMMAND nvcc --version
    RESULT_VARIABLE NVCC_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )

  if(NOT NVIDIA_SMI_RESULT EQUAL 0)
    message(FATAL_ERROR "NVIDIA GPU driver was not detected or nvidia-smi command not available. Aborting GPU build.")
  endif()

  if(NOT NVCC_RESULT EQUAL 0)
    message(FATAL_ERROR "NVIDIA GPU compiler was not detected or nvcc --version command not available. Aborting GPU build.")
  endif()

  find_library(CUDNN_LIB
    NAMES libcudnn.so libcudnn.so.8 libcudnn.so.7
    PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
    DOC "Path to cuDNN library"
  )

  find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    PATHS /usr/local/cuda/include /usr/include
    DOC "Path to cuDNN headers"
  )

  if(NOT (CUDNN_LIB AND CUDNN_INCLUDE_DIR))
    message(FATAL_ERROR "cuDNN not found. Ensure that cuDNN is installed and available in the system paths. Aborting GPU build.")
  endif()

  # If all checks are passed, we try to use the GPU version of ONNXRuntime
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    PATHS "${ONNXRUNTIME_PATH}/lib"
    NO_DEFAULT_PATH
  )

  if(ONNXRUNTIME_LIB)
    message(STATUS "Using GPU version of ONNXRuntime")
    add_definitions(-DUSE_CUDA)
  else()
    message(FATAL_ERROR "GPU version of ONNXRuntime not found. Aborting GPU build.")
  endif()

endif()

# If CPU is selected
if(BUILD_TARGET STREQUAL "CPU")
  message(STATUS "Using CPU version of ONNXRuntime.")
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    PATHS "${ONNXRUNTIME_PATH}/lib"
    NO_DEFAULT_PATH
  )

  if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "CPU version of ONNXRuntime library not found. Please check the library paths or download them from official site:\n
                          https://github.com/microsoft/onnxruntime/releases.")
  endif()
endif()

if(WIN32)
  set(OPENVINO_PATH "C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime" CACHE STRING "OpenVino runtime path")
  set(OPENVINO_LIB_PATH ${OPENVINO_PATH}/lib/intel64/Release CACHE STRING "OpenVino runtime lib path")
else()  
  set(OPENVINO_PATH "/opt/intel/openvino_2024/runtime" CACHE STRING "OpenVino runtime path")
  set(OPENVINO_LIB_PATH ${OPENVINO_PATH}/lib/intel64 CACHE STRING "OpenVino runtime lib path")
endif()

find_library(OPENVINO_LIB
  NAMES openvino_c libopenvino_c
  PATHS ${OPENVINO_LIB_PATH}
  NO_DEFAULT_PATH
)

if(NOT OPENVINO_LIB)
  message(FATAL_ERROR "OpenVino lib is not found at ${OPENVINO_LIB_PATH}")
endif()

# Path for tokenizers-cpp
set(TOKENIZER_CPP_PATH "./tokenizers-cpp")

# Path for cJSON
set(CJSON_PATH "./cJSON")

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add tokenizers-cpp as a subdirectory
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers)

# Add cJSON as a subdirectory
add_subdirectory(${CJSON_PATH} cJSON)

# Add shared library
add_library(GLiClass SHARED
              src/gliclass_api.c
              src/parallel_processor.c
              src/postprocessor.c
              src/model.c
              src/tokenizer.c
              src/preprocessor.c
              src/read_data.c)

# Definitions for DLL export
target_compile_definitions(GLiClass PRIVATE GLICLASS_EXPORTS)

# Public headers (for installation)
set_target_properties(GLiClass PROPERTIES
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/GLiClass/gliclass_api.h")

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PUBLIC
              ${CMAKE_CURRENT_SOURCE_DIR}/include
              ${CMAKE_CURRENT_SOURCE_DIR}
              ${TOKENIZER_CPP_PATH}/include 
              ${ONNXRUNTIME_PATH}/include
              ${OPENVINO_PATH}/include
              ${CJSON_PATH})

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass 
              tokenizers_cpp 
              cjson 
              ${ONNXRUNTIME_LIB}
              ${OPENVINO_LIB}
              OpenMP::OpenMP_C)

# Install DLL and LIB files
install(TARGETS GLiClass
    RUNTIME DESTINATION bin  # DLL
    LIBRARY DESTINATION lib  # Shared libraries
    ARCHIVE DESTINATION lib  # Static libraries
    PUBLIC_HEADER DESTINATION include/GLiClass  # Public headers
)

# Install headers inside include/ preserving directory structure
install(DIRECTORY 
  ${CMAKE_CURRENT_SOURCE_DIR}/include/ 
  DESTINATION include
)
