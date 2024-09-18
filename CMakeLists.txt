cmake_minimum_required(VERSION 3.18)
project(GLiClass C CXX)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path for tokenizers-cpp
set(TOKENIZER_CPP_PATH "./tokenizers-cpp")

# Add tokenizers-cpp as a subdirectory
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

# # Include headers for tokenizers
# include_directories(tokenizers-cpp/include)

# Add the executable (your main C file)
add_executable(GLiClass main.c)

# Include directories with tokenizers-cpp header files
target_include_directories(GLiClass PRIVATE ${TOKENIZER_CPP_PATH}/include)

# Link tokenizers-cpp libraries
target_link_libraries(GLiClass tokenizers_cpp)

