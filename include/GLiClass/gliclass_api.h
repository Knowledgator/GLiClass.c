#ifndef GLICLASS_API_H_
#define GLICLASS_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"

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
    GLiClassTokensInfo* info
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
    GLiClassTokensInfo* info
);

#ifdef __cplusplus
}
#endif

#endif