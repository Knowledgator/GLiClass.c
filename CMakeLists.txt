cmake_minimum_required(VERSION 3.18)
project(GLiClass C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the headers
include_directories(include)

# Add option to choose between CPU and GPU
set(BUILD_TARGET "CPU" CACHE STRING "Choose the build target: CPU or GPU")
set_property(CACHE BUILD_TARGET PROPERTY STRINGS "CPU" "GPU")

message(STATUS "Build target set to: ${BUILD_TARGET}")

############################################ ONNX ###########################################

# Path to ONNXRuntime GPU
set(ONNXRUNTIME_GPU_PATH "./onnxruntime-linux-x64-gpu-1.19.2")
# Path to ONNXRuntime CPU
set(ONNXRUNTIME_CPU_PATH "./onnxruntime-linux-x64-1.19.2")

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
    PATHS "${ONNXRUNTIME_GPU_PATH}/lib"
    NO_DEFAULT_PATH
  )

  if(ONNXRUNTIME_LIB)
    set(ONNXRUNTIME_PATH ${ONNXRUNTIME_GPU_PATH})
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
    PATHS "${ONNXRUNTIME_CPU_PATH}/lib"
    NO_DEFAULT_PATH
  )

  if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "CPU version of ONNXRuntime library not found. Please check the library paths or download them from official site:\n
                          https://github.com/microsoft/onnxruntime/releases.")
  else()
    set(ONNXRUNTIME_PATH ${ONNXRUNTIME_CPU_PATH})
    message(STATUS "Using CPU version of ONNXRuntime")
  endif()
endif()

# Include directories for ONNXRuntime
include_directories(
  "${ONNXRUNTIME_PATH}/include/*"
  "${ONNXRUNTIME_PATH}/include/onnxruntime"
  "${ONNXRUNTIME_PATH}/include/onnxruntime/core/session"
)

###########################################

# Path for tokenizers-cpp
set(TOKENIZER_CPP_PATH "./tokenizers-cpp")

# Path for cJSON
set(CJSON_PATH "./cJSON")

# Add tokenizers-cpp as a subdirectory
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

# Add cJSON as a subdirectory
add_subdirectory(${CJSON_PATH} cJSON EXCLUDE_FROM_ALL)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add the executable (your main C file)
add_executable(GLiClass
                src/parallel_processor.c
                src/postprocessor.c
                src/model.c
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                test.c)

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PRIVATE ${TOKENIZER_CPP_PATH}/include ${CJSON_PATH} ${ONNXRUNTIME_PATH}/include)

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass tokenizers_cpp cjson ${ONNXRUNTIME_LIB} OpenMP::OpenMP_C)
