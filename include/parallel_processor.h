#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdio.h>
#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "tokenizers_c.h"
#include "configs.h"

void parallel_preprocess(OrtValue*** input_ids_tensors, OrtValue*** attention_mask_tensors,
                         const char** texts, const char*** labels, size_t* num_labels,
                         size_t num_texts, bool same_labels, bool prompt_first,
                         TokenizerHandle tokenizer_handler);

void parallel_postprocess(OrtValue** output_tensors, const OrtApi* g_ort,
                          bool same_labels, const char*** labels, size_t* num_labels,
                          size_t num_labels_size, float threshold, size_t num_texts,
                          const char** texts, const char* classification_type);

#endif // PARALLEL_PROCESSOR_H