#ifndef GLICLASS_H_
#define GLICLASS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "GLiClass/gliclass_common.h"

#ifdef USE_ONNX
#include "GLiClass/gliclass_ort.h"
#endif

#ifdef USE_OPENVINO
#include "GLiClass/gliclass_ov.h"
#endif

#include "GLiClass/gliclass_api.h"

#ifdef __cplusplus
}
#endif

#endif