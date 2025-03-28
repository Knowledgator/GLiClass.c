#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stddef.h>
#include <stdbool.h>
#include "GLiClass/gliclass_common.h"

/**
 * Prepares inputs for further processing by combining texts with their corresponding labels.
 * Each input will be formatted according to whether the labels should be included first (prompt_first)
 * and whether the same labels apply to all texts.
 *
 * @param texts An array of input texts.
 * @param labels An array of labels (or array of arrays if same_labels is false).
 * @param num_texts The number of texts to process.
 * @param num_labels An array of size_t, indicating the number of labels for each text.
 * @param same_labels If true, all texts share the same labels; otherwise, each text has its own labels.
 * @param prompt_first If true, labels are added before the text; otherwise, they are appended after the text.
 * @return A dynamically allocated array of strings, where each string contains the prepared input.
 *         The caller is responsible for freeing the memory.
 */
GLiClassStatus* prepare_inputs(
    GLiClassModelConfig* model_config,
    const GLiClassInferenceConfig* config,
    const char* texts[], 
    const size_t num_texts,
    const char** labels[], 
    const size_t num_labels[],
    const bool same_labels,
    char*** inputs
);

/**
 * Prepares a single input by combining a text with its labels, following a specific format.
 * If prompt_first is true, the labels are prepended to the text, otherwise they are appended.
 * Labels are formatted in lower case and are prefixed by "<<LABEL>>".
 *
 * @param text The input text to be prepared.
 * @param labels An array of labels corresponding to the text.
 * @param num_labels The number of labels to process for the text.
 * @param prompt_first If true, labels are added before the text; otherwise, they are appended after the text.
 * @return A dynamically allocated string containing the prepared input. The caller is responsible for freeing the memory.
 */
GLiClassStatus* prepare_input(
    const char* text, 
    const char** labels,
    size_t num_labels,
    bool prompt_first,
    bool add_prefix_space,
    char** input
);

void free_prepared_inputs(char** prepared_inputs, size_t num_texts);

#endif // PREPROCESSOR_H
