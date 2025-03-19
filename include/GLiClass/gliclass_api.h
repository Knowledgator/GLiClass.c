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
#include "tokenizers_c.h"

GLICLASS_API extern const OrtApi* g_ort;

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

typedef struct GLiClassSession {
    const GLiClassModelConfig* model_config;
    TokenizerHandle tokenizer;
    OrtSession* session;
    OrtEnv* env;
    bool use_mutex;
} GLiClassSession;

// Struct to hold inference results
typedef struct GLiClassResult {
    char* label;   // Predicted label
    float score;   // Confidence score
} GLiClassResult;

GLICLASS_API const OrtApi* gliclass_initialize_ort_api();

bool gliclass_create_inference_config(
    size_t batch_size,
    size_t max_length,
    float threshold,
    char* classification_type,
    bool add_prefix_space,
    GLiClassInferenceConfig* config
);

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
    const size_t num_threads,
    const bool use_mutex
);

// Initialize ORT environment
GLICLASS_API OrtEnv* gliclass_create_ort_env(const char* env_name);

/**
 * Creates and initializes an ONNX Runtime session from a model file.
 * 
 * @param env A pointer to the ONNX Runtime environment.
 * @param model_path The file path to the ONNX model.
 * @param num_threads The number of threads to use for inference (CPU only).
 * @return A pointer to the OrtSession if successful, or NULL if an error occurs.
 */
GLICLASS_API OrtSession* gliclass_create_ort_session_cpu_default(
    OrtEnv* env, const char* model_path, const int num_threads
);
GLICLASS_API OrtSession* gliclass_create_ort_session_openvino(
    OrtEnv* env, const char* model_path, const int num_threads, const char* device_type
);
GLICLASS_API OrtSession* gliclass_create_ort_session_cuda(
    OrtEnv* env, const char* model_path, const int num_threads, const int device_id
);

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
    const bool use_mutex,
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
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
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
    const GLiClassInferenceConfig* config,
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