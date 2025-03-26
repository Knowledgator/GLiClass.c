#ifndef GLICLASS_OV_H_
#define GLICLASS_OV_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"
#include <openvino/c/openvino.h>

typedef struct GLiClassSessionOpenVino {
    ov_core_t* core;
    ov_compiled_model_t* model;
} GLiClassSessionOpenVino;

GLICLASS_API GLiClassSessionOpenVino* gliclass_init_openvino_runtime(
    const char* model_path,
    // const char* bin_path,
    const char* model_config_path,
    const char* tokenizer_path,
    const int num_threads,
    const char* device_type,
    const bool use_mutex
);

GLICLASS_API bool gliclass_infer_openvino(
    GLiClassSessionOpenVino* session,
    const GLiClassInferenceConfig* config,
    const char* input_text,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult* out_results[],
    size_t* out_num_results,
    GLiClassTokensInfo* info
);

GLICLASS_API void gliclass_cleanup_openvino(GLiClassSessionOpenVino* session);

#ifdef __cplusplus
}
#endif

#endif