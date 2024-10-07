#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdio.h>
#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "tokenizers_c.h"
#include "configs.h"

void parallel_preprocess(char** texts, char*** labels, size_t* num_labels, size_t num_texts,
                        bool same_labels, bool prompt_first, TokenizerHandle tokenizer_handler,
                        OrtValue** input_ids_tensors, OrtValue** attention_mask_tensors);

void parallel_postprocess(OrtValue** output_tensors, size_t num_batches, size_t num_texts,
                         char** texts, char*** labels, size_t* num_labels,
                         bool same_labels, size_t num_labels_size, const char* classification_type);

#endif // PARALLEL_PROCESSOR_H