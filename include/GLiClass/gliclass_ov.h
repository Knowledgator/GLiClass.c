#ifndef GLICLASS_OV_H_
#define GLICLASS_OV_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"

typedef struct ov_core ov_core_t;
typedef struct ov_compiled_model ov_compiled_model_t;
typedef struct ov_tensor ov_tensor_t;

typedef struct GLiClassOpenVinoSession {
    ov_core_t* core;
    ov_compiled_model_t* model;
} GLiClassOpenVinoSession;

GLICLASS_API GLiClassStatus* gliclass_openvino_init(
    const char* model_path,
    const int num_threads,
    const char* device_type,
    GLiClassProviderAPI** provider
);

#ifdef __cplusplus
}
#endif

#endif