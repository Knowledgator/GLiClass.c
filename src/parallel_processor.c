#include "parallel_processor.h"
#include "preprocessor.h"
#include "postprocessor.h"
#include "model.h"
#include "configs.h"
#include <omp.h>

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
) {
    #pragma omp parallel
    {   
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
            size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;
            
            #pragma omp critical
            {
                printf("Thread %d processing batch %zu to %zu\n", omp_get_thread_num(), i, i + current_batch_size - 1);
            }

            // Select batch data
            const char** batch_texts = (const char**)&texts[i];
            const char*** batch_labels = (const char***)(same_labels ? (void*)labels : (void*)&labels[i]);
            size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

            // Prepare inputs
            const char** prepared_inputs = prepare_inputs(batch_texts, batch_labels, current_batch_size, 
                                                        batch_num_labels, same_labels, prompt_first);
            if (prepared_inputs == NULL) {
                fprintf(stderr, "prepare_inputs returned NULL for batch %zu to %zu\n", 
                        i, i + current_batch_size - 1);
                free_prepared_inputs((char**)prepared_inputs, current_batch_size);
                continue;
            }

            // Tokenize
            TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, 
                                                      current_batch_size, MAX_LENGTH);

            // Prepare tensors
            OrtValue* input_ids_tensor = NULL;
            OrtValue* attention_mask_tensor = NULL;    

            if (prepare_input_tensors(&tokenized, &input_ids_tensor, &attention_mask_tensor) != 0) {
                #pragma omp critical
                {
                    fprintf(stderr, "Error: Failed to prepare input tensors for batch %zu to %zu\n", 
                            i, i + current_batch_size - 1);
                }
                free_prepared_inputs((char**)prepared_inputs, current_batch_size);
                free_tokenized_inputs(&tokenized);
                continue;
            }

            // Run inference
            OrtValue* output_tensor = run_inference(session, input_ids_tensor, attention_mask_tensor);
            if (output_tensor == NULL) {
                #pragma omp critical
                {
                    fprintf(stderr, "Model inference error for batch %zu to %zu\n", 
                            i, i + current_batch_size - 1);
                }
                ort->ReleaseValue(input_ids_tensor);
                ort->ReleaseValue(attention_mask_tensor);
                free_prepared_inputs((char**)prepared_inputs, current_batch_size);
                free_tokenized_inputs(&tokenized);
                continue;
            }

            // Process output
            process_output_tensor(output_tensor, ort, same_labels, batch_labels, batch_num_labels, 
                                num_labels_size, threshold, current_batch_size, batch_texts, 
                                classification_type);

            // Cleanup
            free_prepared_inputs((char**)prepared_inputs, current_batch_size);
            free_tokenized_inputs(&tokenized);
            ort->ReleaseValue(input_ids_tensor);
            ort->ReleaseValue(attention_mask_tensor);
            ort->ReleaseValue(output_tensor);
        }
    }
  
    return 0;
}
