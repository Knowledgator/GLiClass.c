cmake_minimum_required(VERSION 3.18)
project(GLiClass C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the headers
include_directories(include)

############################################ ONNNX ###########################################
# Path to ONNXRuntime GPU
set(ONNXRUNTIME_GPU_PATH "./onnxruntime-linux-x64-gpu-1.19.2")
# Path to ONNXRuntime CPU
set(ONNXRUNTIME_CPU_PATH "./onnxruntime-linux-x64-1.19.2")

# Try to find GPU version first
find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime libonnxruntime
  PATHS "${ONNXRUNTIME_GPU_PATH}/lib"
  NO_DEFAULT_PATH
)
if(NOT ONNXRUNTIME_LIB)
  # If GPU version not found, try CPU version
  find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    PATHS "${ONNXRUNTIME_CPU_PATH}/lib"
    NO_DEFAULT_PATH
  )
  if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Neither GPU nor CPU version of ONNXRuntime library found. Please check the library paths or download them from oficial site:\n
                          https://github.com/microsoft/onnxruntime/releases.")
  else()
    set(ONNXRUNTIME_PATH ${ONNXRUNTIME_CPU_PATH})
    message(WARNING "Using CPU version of ONNXRuntime")
  endif()
else()
  set(ONNXRUNTIME_PATH ${ONNXRUNTIME_GPU_PATH})
  message(WARNING "Using GPU version of ONNXRuntime")
  add_definitions(-DUSE_CUDA)
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


# Add the executable (your main C file)
add_executable(GLiClass
                src/postprocessor.c
                src/model.c
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                main.c)

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PRIVATE ${TOKENIZER_CPP_PATH}/include ${CJSON_PATH} ${ONNXRUNTIME_PATH}/include)

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass tokenizers_cpp cjson ${ONNXRUNTIME_LIB})

