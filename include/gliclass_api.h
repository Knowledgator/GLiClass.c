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

GLICLASS_API extern const OrtApi* g_ort;

typedef struct {
    bool prompt_first;
} ModelConfig;

typedef struct {
    size_t batch_size;
    size_t max_length;
    float threshold;
    char* classification_type;
    bool add_prefix_space;
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

#ifdef _WIN32
GLICLASS_API wchar_t* convert_path(const char* path);
#endif

GLICLASS_API bool initialize_ort_api();

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
    const InferenceConfig* inference_config,
    const size_t num_threads
);

// Initialize ORT environment
GLICLASS_API OrtEnv* create_ort_env(const char* env_name);

GLICLASS_API OrtSession* create_ort_session_with_openvino(OrtEnv* env, const char* model_path, const int num_threads, const char* device_type);

/**
 * Initialize GLiClass model and tokenizer
 * @param model_path Path to ONNX model
 * @param tokenizer_path Path to tokenizer file
 * @param num_threads Number of threads
 * @return GLiClassSession handle, or NULL on error
 */
GLICLASS_API GLiClassSession* gliclass_init_custom_ort(
    const char* model_config_path,
    const char* tokenizer_path, 
    const InferenceConfig* inference_config,
    OrtSession* session
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
    GLiClassResult** out_results[],
    size_t* out_num_results[],
    size_t* out_num_results_size
);

/**
 * Free result array returned by gliclass_classify
 */
GLICLASS_API void gliclass_free_results(GLiClassResult* results, size_t num_results);

GLICLASS_API void gliclass_free_results_batch(GLiClassResult** results, size_t* num_results, size_t num_results_size);
/**
 * Cleanup session
 */
GLICLASS_API void gliclass_cleanup(GLiClassSession* session);

GLICLASS_API void gliclass_cleanup_custom_ort(GLiClassSession* session);

#ifdef __cplusplus
}
#endif

#endif