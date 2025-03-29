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
    
#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>
#ifdef _WIN32
    #include <windows.h>
    typedef BOOLEAN WIN_BOOLEAN;
#endif
#include "tokenizers_c.h"

#define GLICLASS_STATUS_MESSAGE_SIZE 1024

typedef enum GLiClassStatusCode {
    GC_MEMORY_ERROR,
    GC_LOGICAL_ERROR,
    GC_INFERENCE_ERROR,
    GC_PROVIDER_ERROR,
    GC_FILE_ERROR
} GLiClassStatusCode;

typedef struct GLiClassStatus {
    GLiClassStatusCode code;
    char msg[GLICLASS_STATUS_MESSAGE_SIZE];
} GLiClassStatus;

typedef enum GLiClassProvider {
    GC_ONNX,
    GC_ONNX_OPENVINO,
    GC_OPENVINO
} GLiClassProvider;

typedef enum GLiClassDevice {
    GC_NPU = -3,
    GC_CPU = -2,
    GC_GPU = -1,
    GC_GPU_0 = 0,
    GC_GPU_1,
    GC_GPU_2,
    GC_GPU_3,
    GC_GPU_4,
    GC_GPU_5,
    GC_GPU_6,
    GC_GPU_7,
    GC_GPU_8
} GLiClassDevice;

typedef struct GLiClassModelConfig {
    bool prompt_first;
} GLiClassModelConfig;

/**
 * Structure to store tokenized data for a batch of inputs.
 * 
 * Contains input IDs, token type IDs, and attention masks for each tokenized input.
 * Also includes the batch size (number of texts) and the sequence length (max tokens per text).
 */
typedef struct TokenizedInputs {
    int64_t** input_ids;        /**< Array of token IDs for each input text. */
    int64_t** token_type_ids;   /**< Array of token type IDs for each input text. */
    int64_t** attention_mask;   /**< Array indicating which tokens are actual tokens (1) and which are padding (0). */
    size_t batch_size;      /**< Number of input texts in the batch. */
    size_t seq_length;       /**< Maximum sequence length for the input texts. */
} TokenizedInputs;

/**
 * Structure to store tokenized data for a single input.
 * 
 * Contains input IDs, token type IDs, and attention masks for the input.
 * Also includes the sequence length (max tokens per text).
 */
typedef struct TokenizedInput {
    int64_t* input_ids;        /**< Array of token IDs for each input text. */
    int64_t* token_type_ids;   /**< Array of token type IDs for each input text. */
    int64_t* attention_mask;   /**< Array indicating which tokens are actual tokens (1) and which are padding (0). */
    size_t seq_length;       /**< Maximum sequence length for the input texts. */
} TokenizedInput;

typedef struct GLiClassProviderAPI GLiClassProviderAPI;

typedef struct GLiClassSession {
    GLiClassModelConfig* model_config;
    TokenizerHandle tokenizer;
    GLiClassProviderAPI* provider;
    bool use_mutex;
} GLiClassSession;

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

typedef struct GLiClassProviderAPI {
    void* session;
    GLiClassStatus* (*run_inference)(
        GLiClassSession*, 
        const GLiClassInferenceConfig*, 
        const TokenizedInput*, 
        const char**, 
        const size_t,
        GLiClassResult**, 
        size_t*
    );
    GLiClassStatus* (*run_inference_batch)(
        GLiClassSession*,
        const GLiClassInferenceConfig*,
        const TokenizedInputs*,
        const size_t,
        const char***,
        const size_t*,
        const size_t,
        GLiClassResult***,
        size_t**
    );
    void (*cleanup)(void*);
} GLiClassProviderAPI;

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
GLICLASS_API GLiClassStatus* gliclass_create_inference_config(
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

GLICLASS_API void gliclass_free_status(GLiClassStatus* status);

#ifdef __cplusplus
}
#endif

#endif