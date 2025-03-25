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
    /** Used in batch inference. Specifies max batch size */
    size_t batch_size; 
    /** Minimum length in tokens of the input. If input is shorter than this value it will not be processed. */
    size_t min_length; 
    /** Maximum length in tokens of the input. If input is longer than this value it will be truncated to this value. */
    size_t max_length; 
    /** Scores threshold. If score is less than this value it will not be included in results. */
    float threshold; 
    /** One of: "multi-label": Returns all labels with score higher than threshold; "single-label": Return label with the highest score if the score is higher than threshold; */
    char* classification_type; 
    /** Tokenizer preprocessing config. Adds prefix space to labels and text before tokenization. */
    bool add_prefix_space; 
} GLiClassInferenceConfig;

// Struct to hold inference results
typedef struct GLiClassResult {
    char* label;   // Predicted label
    float score;   // Confidence score
} GLiClassResult;

// Struct to hold tokenization info
typedef struct GLiClassTokensInfo {
    bool truncated;   // Input was truncated
    size_t tokens_num;   // Number of processed tokens
} GLiClassTokensInfo;

/** 
 * @param batch_size Used in batch inference. Specifies max batch size
 * @param min_length Minimum length in tokens of the input. 
 * If input is shorter than this value it will not be processed.
 * @param max_length Maximum length in tokens of the input.
 * If input is longer than this value it will be truncated to this value.
 * @param threshold Scores threshold. If score is less than this value it will not be included in results.
 * @param classification_type One of:
 * < "multi-label": Returns all labels with score higher than threshold;
 * < "single-label": Return label with the highest score if the score is higher than threshold;
 * @param add_prefix_space Tokenizer preprocessing config. Adds prefix space to labels and text before tokenization.
 * @param config Output config pointer.
*/
GLICLASS_API bool gliclass_create_inference_config(
    size_t batch_size,
    size_t min_length,
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