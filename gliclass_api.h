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

// Opaque type for GLiClass session handle
typedef struct GLiClassSession GLiClassSession;

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
GLICLASS_API GLiClassSession* gliclass_init(const char* model_path, const char* tokenizer_path, int num_threads);

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
    const char** labels,
    size_t num_labels,
    GLiClassResult** out_results,
    size_t* out_num_results
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