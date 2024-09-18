cmake_minimum_required(VERSION 3.18)
project(GLiClass C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the headers
include_directories(include)

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
                src/tokenizer.c
                src/preprocessor.c
                src/read_data.c
                main.c)

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PRIVATE ${TOKENIZER_CPP_PATH}/include ${CJSON_PATH})

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass tokenizers_cpp cjson)

