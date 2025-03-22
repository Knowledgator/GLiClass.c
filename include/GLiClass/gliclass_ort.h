#ifndef GLICLASS_ORT_H_
#define GLICLASS_ORT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"
#include "onnxruntime_c_api.h"

GLICLASS_API extern const OrtApi* g_ort;

typedef struct GLiClassSession {
    const GLiClassModelConfig* model_config;
    TokenizerHandle tokenizer;
    OrtSession* session;
    OrtEnv* env;
    bool use_mutex;
} GLiClassSession;

GLICLASS_API const OrtApi* gliclass_initialize_ort_api();

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
    const int num_threads,
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
 * @param config Inference config
 * @param text Input text
 * @param labels Array of candidate labels
 * @param num_labels Number of labels
 * @param out_results Array of results (allocated inside function)
 * @param out_num_results Number of results returned
 * @param truncated If the input was truncated during preprocessing
 * (i.e., the number of tokens was reduced to the max_length specified in the inference config).
 * @return true on success, false on error
 */
GLICLASS_API bool gliclass_infer(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results,
    bool* truncated
);

/**
 * Perform batch classification on input texts and labels
 * @param session Initialized session handle
 * @param config Inference config
 * @param text Input text
 * @param labels Array of candidate labels
 * @param num_labels Number of labels
 * @param out_results Array of results (allocated inside function)
 * @param out_num_results Number of results returned for each text
 * @param out_num_result_size Number of results returned for texts
 * @param truncated If the input was truncated during preprocessing
 * (i.e., the number of tokens was reduced to the max_length specified in the inference config),
 * the array represents each processed input (per text-labels pair).
 * @return true on success, false on error
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
    size_t* out_num_results_size,
    bool* truncated[]
);

/**
 * Cleanup session
 */
GLICLASS_API void gliclass_cleanup(GLiClassSession* session);

GLICLASS_API void gliclass_cleanup_custom_ort(GLiClassSession* session);

#ifdef __cplusplus
}
#endif

#endif