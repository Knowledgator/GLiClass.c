#ifndef GLICLASS_ORT_H_
#define GLICLASS_ORT_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"
#include "onnxruntime_c_api.h"

GLICLASS_API extern const OrtApi* g_ort;

typedef struct GLiClassORTSession {
    OrtSession* session;
    OrtEnv* env;
} GLiClassORTSession;

GLICLASS_API const OrtApi* gliclass_ort_initialize_api();

// Initialize ORT environment
GLICLASS_API GLiClassStatus* gliclass_ort_create_env(const char* env_name, OrtEnv** env);

/**
 * Creates and initializes an ONNX Runtime session from a model file.
 * 
 * @param env A pointer to the ONNX Runtime environment.
 * @param model_path The file path to the ONNX model.
 * @param num_threads The number of threads to use for inference (CPU only).
 * @return A pointer to the OrtSession if successful, or NULL if an error occurs.
 */
GLICLASS_API GLiClassStatus* gliclass_ort_cpu_init(
    const char* model_path, const int num_threads, GLiClassProviderAPI** provider
);
GLICLASS_API GLiClassStatus* gliclass_ort_openvino_init(
    const char* model_path, const int num_threads, const char* device_type, GLiClassProviderAPI** provider
);
#ifdef GC_USE_CUDA
GLICLASS_API GLiClassStatus* gliclass_ort_cuda_init(
    const char* model_path, const int num_threads, const int device_id, GLiClassProviderAPI** provider
);
#endif

/**
 * Initialize GLiClass model and tokenizer
 * @param model_path Path to ONNX model
 * @param tokenizer_path Path to tokenizer file
 * @param num_threads Number of threads
 * @return GLiClassSessionORT handle, or NULL on error
 */
GLICLASS_API GLiClassStatus* gliclass_ort_init_custom(
    OrtSession* ort_session, GLiClassProviderAPI** provider
);

#ifdef __cplusplus
}
#endif

#endif