#ifndef GLICLASS_API_H_
#define GLICLASS_API_H_

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
#include "onnxruntime_c_api.h"
#include "tokenizer.h"

extern const OrtApi* g_ort;

typedef struct {
    bool prompt_first;
} ModelConfig;

// #ifndef CONFIGS_H
// #define CONFIGS_H

// #define BATCH_SIZE 8    // Number of texts in one batch for processing by the model
// #define MAX_LENGTH 2048 // Maximum length of tokenized text (number of tokens)
// #define THRESHOLD 0.5f  // Threshold for making a classification decision 
// #define NUM_THREADS 8   // Number of threads for CPU (does not affect GPU performance)

// #endif // CONFIGS_H

typedef struct {
    size_t batch_size;
    size_t max_length;
    size_t cpu_threads;
    float threshold;
    char* classification_type;
} InferenceConfig;

typedef struct {
    const ModelConfig* model_config;
    const InferenceConfig* inference_config; 
    TokenizerHandle tokenizer;
    OrtSession* session;
    OrtEnv* env;
} GLiClassSession;

// Struct to hold inference results
typedef struct {
    char* label;   // Predicted label
    float score;   // Confidence score
} GLiClassResult;

/**
 * Initialize GLiClass model and tokenizer
 * @param model_path Path to ONNX model
 * @param tokenizer_path Path to tokenizer file
 * @param num_threads Number of threads
 * @return GLiClassSession handle, or NULL on error
 */
GLICLASS_API GLiClassSession* gliclass_init(
    const char* model_path, 
    const char* model_config_path,
    const char* tokenizer_path, 
    const InferenceConfig* inference_config
);

/**
 * Perform classification on input text and labels
 * @param session Initialized session handle
 * @param text Input text
 * @param labels Array of candidate labels
 * @param num_labels Number of labels
 * @param out_results Array of results (allocated inside function)
 * @param out_result_count Number of results returned
 * @return 0 on success, non-zero on error
 */
GLICLASS_API bool gliclass_infer(
    GLiClassSession* session,
    const char* input_text,
    const char* labels[],
    size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results
);

/**
 * Perform batch classification on input texts and labels
 * @param session Initialized session handle
 * @param text Input text
 * @param labels Array of candidate labels
 * @param num_labels Number of labels
 * @param out_results Array of results (allocated inside function)
 * @param out_result_count Number of results returned
 * @return 0 on success, non-zero on error
 */
GLICLASS_API bool gliclass_infer_batch(
    GLiClassSession* session,
    const char* input_texts[],
    const size_t num_texts,
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size, // TODO: rename
    GLiClassResult*** out_results[],
    size_t* out_num_results[],
    size_t* out_num_results_size
);

/**
 * Free result array returned by gliclass_classify
 */
GLICLASS_API void gliclass_free_results(GLiClassResult* results, size_t num_results);

/**
 * Cleanup session
 */
GLICLASS_API void gliclass_cleanup(GLiClassSession* session);


#ifdef __cplusplus
}
#endif

#endif