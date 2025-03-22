#ifndef GLICLASS_API_H_
#define GLICLASS_API_H_

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


#ifdef __cplusplus
}
#endif

#endif