#include "parallel_processor.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "model.h"
#include "configs.h"
#include <omp.h>

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
 * @param same_labels Flag indicating if all texts share the same set of labels.
 * @param prompt_first Flag indicating if the prompt should be placed before the input text.
 * @param tokenizer_handler Handle for the tokenizer used to tokenize the input texts.
 * @param input_ids_tensors Output array of input ID tensors for each batch.
 * @param attention_mask_tensors Output array of attention mask tensors for each batch.
 */
void parallel_preprocess(char** texts, char*** labels, size_t* num_labels, size_t num_texts,
                        bool same_labels, bool prompt_first, TokenizerHandle tokenizer_handler,
                        OrtValue** input_ids_tensors, OrtValue** attention_mask_tensors) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
        size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;

        // Prepare input data
        const char** batch_texts = (const char**)&texts[i];
        const char*** batch_labels = (const char***)(same_labels ? (void*)labels : (void*)&labels[i]);
        size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

        // Prepare tokens
        const char** prepared_inputs = prepare_inputs(batch_texts, batch_labels, current_batch_size, 
                                                    batch_num_labels, same_labels, prompt_first);
        TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, 
                                                  current_batch_size, MAX_LENGTH);

        // Prepare input tensors
        prepare_input_tensors(&tokenized, &input_ids_tensors[i / BATCH_SIZE], 
                            &attention_mask_tensors[i / BATCH_SIZE]);

        // Clean up memory
        free_prepared_inputs((char**)prepared_inputs, current_batch_size);
        free_tokenized_inputs(&tokenized);
    }
}

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
void parallel_postprocess(OrtValue** output_tensors, size_t num_batches, size_t num_texts,
                         char** texts, char*** labels, size_t* num_labels,
                         bool same_labels, size_t num_labels_size, const char* classification_type) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        size_t current_batch_size = (i == num_batches - 1) ? 
                                  (num_texts - i * BATCH_SIZE) : BATCH_SIZE;

        const char** batch_texts = (const char**)&texts[i * BATCH_SIZE];
        const char*** batch_labels = (const char***)(same_labels ? 
                                   (void*)labels : (void*)&labels[i * BATCH_SIZE]);
        size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i * BATCH_SIZE];

        process_output_tensor(output_tensors[i], g_ort, same_labels, batch_labels, 
                            batch_num_labels, num_labels_size, THRESHOLD,
                            current_batch_size, batch_texts, classification_type);
        
        // Free output tensor after processing
        g_ort->ReleaseValue(output_tensors[i]);
    }
}