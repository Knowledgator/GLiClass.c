#ifndef POSTPROCESSOR_H
#define POSTPROCESSOR_H

#include <stdbool.h>
#include "GLiClass/gliclass_ort.h"

/**
 * Processes the output tensor (logits) and prints the predicted labels and scores based on the given classification type (multi-label or single-label).
 * 
 * @param session A pointer to GLiClass session
 * @param output_tensor A pointer to the OrtValue containing the output logits from the ONNX model.
 * @param labels A 2D array of strings containing the labels for each class.
 * @param num_labels A dynamic array indicating the number of labels for each text.
 * @param out_results Pointer to results
 */
void process_output_tensor(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    OrtValue* output_tensor,
    const char* labels[],
    const size_t num_labels,
    GLiClassResult out_results[],
    size_t* out_num_results
);

/**
 * Processes the output tensor (logits) and prints the predicted labels and scores based on the given classification type (multi-label or single-label).
 * 
 * @param session A pointer to GLiClass session
 * @param output_tensor A pointer to the OrtValue containing the output logits from the ONNX model.
 * @param labels A 2D array of strings containing the labels for each class.
 * @param num_labels A dynamic array indicating the number of labels for each text.
 * @param num_labels_size The number of labels if all texts share the same set.
 * @param batch_id Butch for proccessing
 * @param out_results Pointer to results
 * @param out_num_results Results shape 
 */
void process_output_tensor_batch(
    GLiClassSession* session,
    const GLiClassInferenceConfig* config,
    OrtValue* output_tensor, 
    const char** labels[],
    const size_t* num_labels,
    const size_t num_labels_size,
    const size_t batch_id,
    GLiClassResult* out_results[],
    size_t out_num_results[]
);

#endif // POSTPROCESSOR_H
