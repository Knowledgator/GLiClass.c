#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdio.h>
#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "tokenizers_c.h"
#include "configs.h"

// Function to process data in parallel batches
int process_batches_parallel(
    const char** texts,
    size_t num_texts,
    char*** labels,
    size_t* num_labels,
    size_t num_labels_size,
    bool same_labels,
    bool prompt_first,
    const char* classification_type,
    TokenizerHandle tokenizer_handler,
    OrtSession* session,
    const OrtApi* ort,
    float threshold
);

#endif // PARALLEL_PROCESSOR_H