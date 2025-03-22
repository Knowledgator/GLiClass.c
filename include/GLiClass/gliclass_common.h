#ifndef GLICLASS_COMMON_H_
#define GLICLASS_COMMON_H_

#ifdef _WIN32
    #ifdef GLICLASS_EXPORTS
        #define GLICLASS_API __declspec(dllexport)
    #else
        #define GLICLASS_API __declspec(dllimport)
    #endif
#else
    #define GLICLASS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif
    
#include <stddef.h>
#include <stdbool.h>
#ifdef _WIN32
    #include <windows.h>
    typedef BOOLEAN WIN_BOOLEAN;
#endif
#include "tokenizers_c.h"

typedef struct GLiClassModelConfig {
    bool prompt_first;
} GLiClassModelConfig;

typedef struct GLiClassInferenceConfig {
    size_t batch_size;
    size_t max_length;
    float threshold;
    char* classification_type;
    bool add_prefix_space;
} GLiClassInferenceConfig;

// Struct to hold inference results
typedef struct GLiClassResult {
    char* label;   // Predicted label
    float score;   // Confidence score
} GLiClassResult;

GLICLASS_API bool gliclass_create_inference_config(
    size_t batch_size,
    size_t max_length,
    float threshold,
    char* classification_type,
    bool add_prefix_space,
    GLiClassInferenceConfig* config
);

/**
 * Free result array returned by gliclass_classify
 */
GLICLASS_API void gliclass_free_results(GLiClassResult* results, size_t num_results);

GLICLASS_API void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size);

#ifdef __cplusplus
}
#endif

#endif