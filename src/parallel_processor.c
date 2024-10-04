#include "parallel_processor.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "model.h"
#include "configs.h"
#include <omp.h>

/**
 * @brief Preprocesses text data in parallel, preparing input tensors for model inference.
 *
 * This function tokenizes text data, prepares the inputs for each batch in parallel, and stores the result
 * in the input tensors for use in model inference. It uses OpenMP for parallel execution.
 *
 * @param input_ids_tensors Pointer to an array of OrtValue pointers for storing input IDs tensors.
 * @param attention_mask_tensors Pointer to an array of OrtValue pointers for storing attention mask tensors.
 * @param texts Array of strings containing the texts to be tokenized and processed.
 * @param labels Array of labels for each text; can be the same for all texts if `same_labels` is true.
 * @param num_labels Array containing the number of labels for each text.
 * @param num_texts Total number of texts in the dataset.
 * @param same_labels Boolean flag indicating whether the same labels are used for all texts.
 * @param prompt_first Boolean flag indicating if the prompt should be prepended to the texts during preprocessing.
 * @param tokenizer_handler A handle to the tokenizer used for tokenizing the input texts.
 */
void parallel_preprocess(OrtValue*** input_ids_tensors, OrtValue*** attention_mask_tensors,
                         const char** texts, const char*** labels, size_t* num_labels,
                         size_t num_texts, bool same_labels, bool prompt_first,
                         TokenizerHandle tokenizer_handler) {

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
        size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;

        // Prepare input data
        const char** batch_texts = (const char**)&texts[i];
        const char*** batch_labels = (const char***)(same_labels ? (void*)labels : (void*)&labels[i]);
        size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

        // Prepare tokens
        const char** prepared_inputs = prepare_inputs(batch_texts, batch_labels, current_batch_size, batch_num_labels, same_labels, prompt_first);
        TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, current_batch_size, MAX_LENGTH);

        // Prepare input tensors
        prepare_input_tensors(&tokenized, &(*input_ids_tensors)[i / BATCH_SIZE], &(*attention_mask_tensors)[i / BATCH_SIZE]);

        // Clean up memory
        free_prepared_inputs((char**)prepared_inputs, current_batch_size);
        free_tokenized_inputs(&tokenized);
    }
}


/**
 * @brief Postprocesses model outputs in parallel, decoding the predictions for each batch.
 *
 * This function takes the output tensors from model inference, processes them to extract classification results,
 * and performs postprocessing in parallel. The processed results can be labels or scores for each text batch.
 *
 * @param output_tensors Array of OrtValue pointers containing the output tensors from model inference.
 * @param g_ort Pointer to the ONNX Runtime API, used to access the runtime functions for tensor manipulation.
 * @param same_labels Boolean flag indicating whether the same labels are used for all texts.
 * @param labels Array of labels for each text; can be the same for all texts if `same_labels` is true.
 * @param num_labels Array containing the number of labels for each text.
 * @param num_labels_size The size of the label array (total number of possible labels).
 * @param threshold The threshold for deciding binary classification results (if applicable).
 * @param num_texts Total number of texts in the dataset.
 * @param texts Array of strings containing the original input texts, used during postprocessing for reference.
 * @param classification_type String indicating the classification type (e.g., "single-label", "multi-label").
 */
void parallel_postprocess(OrtValue** output_tensors, const OrtApi* g_ort,
                          bool same_labels, const char*** labels, size_t* num_labels,
                          size_t num_labels_size, float threshold, size_t num_texts,
                          const char** texts, const char* classification_type) {

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
        size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;

        const char** batch_texts = (const char**)&texts[i];
        const char*** batch_labels = (const char***)(same_labels ? (void*)labels : (void*)&labels[i]);
        size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

        // Process model output
        process_output_tensor(output_tensors[i / BATCH_SIZE], g_ort, same_labels, batch_labels, batch_num_labels, num_labels_size, threshold,
                              current_batch_size, batch_texts, classification_type);
    }
}