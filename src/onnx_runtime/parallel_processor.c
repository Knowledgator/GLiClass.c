#include "parallel_processor.h"

#include "../preprocessor.h"
#include "postprocessor.h"
#include "model.h"

size_t get_batch_size(
    const size_t batch_idx,
    const size_t num_batches, 
    const size_t num_texts, 
    const size_t batch_size
) {
    return (batch_idx == num_batches - 1) ? (num_texts - batch_idx * batch_size) : batch_size;
}

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
) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        const size_t batch_size = get_batch_size(
            i, num_batches, num_texts, config->batch_size
        );

        // Prepare input data
        bool same_labels = num_labels_size == 1;
        const char** batch_texts = texts + i*batch_size;
        const char*** batch_labels = same_labels ? labels : labels + i*batch_size;
        const size_t* batch_num_labels = same_labels ? num_labels : num_labels + i*batch_size;

        // Prepare tokens
        const char** prepared_inputs = prepare_inputs(
            session->model_config,
            config,
            batch_texts,
            batch_size,
            batch_labels, 
            batch_num_labels, 
            same_labels
        );
        TokenizedInputs tokenized = tokenize_inputs(
            session->tokenizer, 
            (const char**)prepared_inputs, 
            batch_size,
            config->min_length,
            config->max_length
        );
        *truncated = tokenized.truncated;

        // Prepare input tensors
        prepare_input_tensors(
            &tokenized,
            &input_ids_tensors[i / config->batch_size], 
            &attention_mask_tensors[i / config->batch_size]
        );

        // Clean up memory
        free_prepared_inputs((char**)prepared_inputs, batch_size);
        free_tokenized_inputs(&tokenized);
    }
}

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
) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        const size_t batch_size = get_batch_size(
            i, num_batches, num_texts, config->batch_size
        );
        const char*** batch_labels = (
            same_labels ? labels : (labels + i*config->batch_size)
        );
        const size_t* batch_num_labels = (
            same_labels ? num_labels : (num_labels + i*config->batch_size)
        );
        const size_t batch_num_labels_size = same_labels ? num_labels_size: batch_size;

        process_output_tensor_batch(
            session,
            config,
            output_tensors[i], 
            batch_labels, 
            batch_num_labels, 
            batch_num_labels_size, 
            i, 
            out_results,
            out_num_results
        );
        
        // Free output tensor after processing
        g_ort->ReleaseValue(output_tensors[i]);
    }
    free(output_tensors);
}