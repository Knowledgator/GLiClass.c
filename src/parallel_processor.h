#ifndef PARALLEL_PROCESSOR_H
#define PARALLEL_PROCESSOR_H

#include <stdio.h>
#include <stdbool.h>
#include "onnxruntime_c_api.h"
#include "GLiClass/gliclass_api.h"
#include "tokenizers_c.h"

/**
 * @brief Preprocesses a batch of texts and labels in parallel.
 *
 * This function takes an array of texts and labels, and preprocesses them in parallel
 * using OpenMP. It prepares the input data, tokenizes it, and creates the necessary
 * input tensors for the model.
 *
 * @param texts Array of input texts.
 * @param labels Array of label arrays for each text. If same_labels is true, this is a single set of labels.
 * @param num_labels Array containing the number of labels for each text. If same_labels is true, this is a single value.
 * @param num_texts Number of texts to be processed.
 * @param prompt_first Flag indicating if the prompt should be placed before the input text.
 * @param tokenizer_handler Handle for the tokenizer used to tokenize the input texts.
 * @param input_ids_tensors Output array of input ID tensors for each batch.
 * @param attention_mask_tensors Output array of attention mask tensors for each batch.
 */
void parallel_preprocess(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    const size_t num_batches,
    const char* texts[], 
    const size_t num_texts,
    const char** labels[], 
    const size_t num_labels[],
    const size_t num_labels_size,
    OrtValue** input_ids_tensors,
    OrtValue** attention_mask_tensors,
    bool** truncated
);

/**
 * @brief Postprocesses the output tensors in parallel.
 *
 * This function takes the output tensors from the model and postprocesses them in parallel
 * using OpenMP. It processes each output tensor to extract the relevant information and
 * updates the provided labels accordingly.
 *
 * @param output_tensors Array of output tensors from the model for each batch.
 * @param num_batches Number of batches processed.
 * @param num_texts Total number of texts processed.
 * @param texts Array of input texts.
 * @param labels Array of label arrays for each text. If same_labels is true, this is a single set of labels.
 * @param num_labels Array containing the number of labels for each text. If same_labels is true, this is a single value.
 * @param same_labels Flag indicating if all texts share the same set of labels.
 * @param num_labels_size Size of the label arrays.
 * @param classification_type Type of classification being performed (e.g., "binary", "multi-class").
 */
void parallel_postprocess(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    OrtValue** output_tensors, 
    const size_t num_batches,
    const size_t num_texts,
    const char** labels[],
    const size_t num_labels[],
    const size_t num_labels_size, 
    const bool same_labels,
    GLiClassResult* out_results[],
    size_t out_num_results[]
);

#endif // PARALLEL_PROCESSOR_H