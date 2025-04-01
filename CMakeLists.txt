cmake_minimum_required(VERSION 3.19)
project(GLiClass C CXX)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Set C standard
if (MSVC)
else()
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  add_compile_options(-fcommon)
endif()

include(FetchContent)

# Add option to choose between CPU and GPU
option(ONNX "Build ONNX runtime inference with default CPU provider" ON)
option(ONNX_CUDA "Build ONNX runtime inference with CUDA provider support" OFF)
option(ONNX_OPENVINO "Build ONNX runtime inference with OpenVino provider support" OFF)
option(OPENVINO "Build Openvino runtime inference" OFF)
option(SKIP_CHECK "Skip environment check" OFF)

if(ONNX_CUDA OR ONNX_OPENVINO)
  SET(ONNX ON)
endif()

if(ONNX_OPENVINO)
  SET(OPENVINO ON)
endif()

set(BUILD_TARGET "")
if(ONNX)
  string(APPEND BUILD_TARGET "ONNX; ")
endif()
if(ONNX_CUDA)
  string(APPEND BUILD_TARGET "ONNX_CUDA; ")
endif()
if(ONNX_OPENVINO)
  string(APPEND BUILD_TARGET "ONNX_OPENVINO; ")
endif()
if(OPENVINO)
  string(APPEND BUILD_TARGET "OPENVINO; ")
endif()  

message(STATUS "Build target set to: ${BUILD_TARGET}")

############################################ ONNX ###########################################

# Path to ONNXRuntime 
set(ONNXRUNTIME_PATH "" CACHE STRING "Onnxruntime path")
set(OPENVINO_PATH "" CACHE STRING "OpenVino runtime path")
set(OPENVINO_LIB_PATH "" CACHE STRING "OpenVino runtime lib path")

# If using ONNX runtime
if(ONNX)
  if ("${ONNXRUNTIME_PATH}" STREQUAL "")
    find_library(ONNXRUNTIME_LIB
      NAMES onnxruntime libonnxruntime
    )
  else()
    find_library(ONNXRUNTIME_LIB
      NAMES onnxruntime libonnxruntime
      PATHS "${ONNXRUNTIME_PATH}/lib"
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNXRuntime library not found. Please check the library paths or download them from official site:\n
            https://github.com/microsoft/onnxruntime/releases.")
  endif()
endif()

if(ONNX_CUDA)
  # Perform checks for GPU, NVIDIA and cuDNN availability
  if(NOT SKIP_CHECK)
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
  endif()
endif()


if(OPENVINO)
  if ("${OPENVINO_PATH}" STREQUAL "")
    find_library(OPENVINO_LIB
      NAMES openvino_c libopenvino_c
    )
  else()
    if ("${OPENVINO_LIB_PATH}" STREQUAL "")
      if(WIN32)
        set(OPENVINO_LIB_PATH ${OPENVINO_PATH}/lib/intel64/Release)
      else()  
        set(OPENVINO_LIB_PATH ${OPENVINO_PATH}/lib/)
      endif()
    endif()
    find_library(OPENVINO_LIB
      NAMES openvino_c libopenvino_c
      PATHS ${OPENVINO_LIB_PATH}
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT OPENVINO_LIB)
    message(FATAL_ERROR "OpenVino lib is not found at ${OPENVINO_LIB_PATH}")
  endif()
endif()

# Tokenizers CPP
if(NOT TARGET tokenizers_cpp)
  FetchContent_Declare(
      tokenizers_cpp
      GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
      GIT_TAG 832b8728983be015f7db7428741ede87f925eddb # latest version 24.02.25
  )
  FetchContent_MakeAvailable(tokenizers_cpp)
endif()

# cJSON
if(NOT TARGET cjson)
  FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG acc76239bee01d8e9c858ae2cab296704e52d916 # latest version 1.7.18
  )
  FetchContent_MakeAvailable(cjson)
endif()

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add shared library
if(ONNX AND OPENVINO)
  add_library(GLiClass SHARED
                src/gliclass_ort.c
                src/onnx_runtime/model.c
                src/onnx_runtime/postprocessor.c
                src/gliclass_ov.c
                src/openvino_runtime/model.c
                src/openvino_runtime/postprocessor.c
                src/gliclass_common.c
                src/gliclass_api.c
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                src/utils.c
                src/error.c)
elseif(ONNX)
  add_library(GLiClass SHARED
                src/gliclass_ort.c
                src/onnx_runtime/model.c
                src/onnx_runtime/postprocessor.c
                src/gliclass_common.c
                src/gliclass_api.c
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                src/utils.c
                src/error.c)
elseif(OPENVINO)
  add_library(GLiClass SHARED
                src/gliclass_ov.c
                src/openvino_runtime/model.c
                src/openvino_runtime/postprocessor.c
                src/gliclass_common.c
                src/gliclass_api.c
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                src/utils.c
                src/error.c)
else()
  message(FATAL_ERROR "Non of provided runtimes was selected")
endif()

# Definitions for DLL export
target_compile_definitions(GLiClass PRIVATE GLICLASS_EXPORTS)
if(ONNX)
  target_compile_definitions(GLiClass PRIVATE GC_USE_ONNX)
endif()
if(ONNX_CUDA)
  target_compile_definitions(GLiClass PRIVATE GC_USE_CUDA)
endif()
if(OPENVINO)
  target_compile_definitions(GLiClass PRIVATE GC_USE_OPENVINO)
  target_compile_definitions(GLiClass PRIVATE OPENVINO_STATIC_LIBRARY)
endif()

# Public headers (for installation)
set_target_properties(GLiClass PROPERTIES
  PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/GLiClass/gliclass.h")

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PUBLIC
              ${CMAKE_CURRENT_SOURCE_DIR}/include
              ${CMAKE_CURRENT_SOURCE_DIR})

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass
              tokenizers_cpp 
              cjson
              OpenMP::OpenMP_C)

if(ONNX)
  if (NOT "${ONNXRUNTIME_PATH}" STREQUAL "")
    target_include_directories(GLiClass PUBLIC ${ONNXRUNTIME_PATH}/include)
  endif()
  target_link_libraries(GLiClass ${ONNXRUNTIME_LIB})
endif()
if(OPENVINO)
  if (NOT "${OPENVINO_PATH}" STREQUAL "")
    target_include_directories(GLiClass PUBLIC ${OPENVINO_PATH}/include)
  endif()
  target_link_libraries(GLiClass ${OPENVINO_LIB})
endif()

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
