#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include "onnxruntime_c_api.h"

void process_output_tensor(OrtValue* output_tensor, const OrtApi* g_ort);

#endif // POSTPROCESSOR_H
