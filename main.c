#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"
#include "onnxruntime_c_api.h"
#include <omp.h>

// Project includes (folder include)
#include "postprocessor.h"
#include "model.h"
#include "tokenizer.h" 
#include "preprocessor.h"
#include "read_data.h"
#include "paths.h"
#include "configs.h"

// Ini variables for data
char** texts = NULL;                // Array of strings containing texts to classify
size_t num_texts = 0;               // Number of texts in the 'texts' array
char*** labels = NULL;              // An array of labels for each text; there can be multiple labels for each text
size_t* num_labels = NULL;          // Array containing the number of tags for each text
size_t num_labels_size = 0;         // Total size of the array of labels 
bool same_labels = false;           // Flag indicating whether the same labels are used for all texts
char* classification_type = NULL;   // Classification type (e.g. single-label, multi-label)

const OrtApi* g_ort = NULL;         // Global pointer to ONNX Runtime API for performing model inference

/**
 * Main function that runs the text classification model using ONNX Runtime.
 * It reads input data from a JSON file, preprocesses the texts, tokenizes them, runs inference using the ONNX model,
 * and processes the output logits to print the classification results. It supports multi-threading using OpenMP.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments. argv[1] should be the path to the input JSON file.
 * @return 0 if successful, or 1 if an error occurs (e.g., invalid arguments or failed initialization).
 */
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s /path/to/your_data.json [prompt_first: true/false]\n", argv[0]);
        printf("NOTE: use this option only if you sure that all required model parts are initialized correctly\n\n");
        printf("Recomended option\n");
        printf("Usage: ./run_GLiClass.sh knowledgator/gliclass-small-v1.0 /path/to/your_data.json\n");
        printf("This option will automaticly set up prompt_first for you\n");
        return 1;
    }
    // reading data from json file
    char* json_string = read_file(argv[1]);
    if (!json_string) {
        return 1;
    }
    bool prompt_first = string_to_bool(argv[2]);

    // clock_t start, end;
    // double cpu_time_used;
    double start_time, end_time, cpu_time_used;

    ///////////// Prepare inputs /////////////
    parse_json(json_string, &texts, &num_texts, &labels, &num_labels, &num_labels_size, &same_labels, &classification_type);
    printf("DONE: parse_json;\n");
    if (classification_type == NULL){
        printf("classification type is not provided\n");
        return 1;
    }
    ///////////// intializing part /////////////
    TokenizerHandle tokenizer_handler = create_tokenizer(TOKENIZER_PATH);
    if (!tokenizer_handler) {
        return 1; // This error is created in create_tokenizer
    }
    printf("DONE: create_tokenizer;\n");  

    initialize_ort_api();
    printf("DONE: initialize_ort_api;\n");

    OrtEnv* env = initialize_ort_environment();
    if (env == NULL) {
        fprintf(stderr, "Error: Failed to initialize ONNX Runtime.\n");
        return -1;
    }
    printf("DONE: initialize_ort_environment;\n");

    OrtSession* session = create_ort_session(env, MODEL_PATH, NUM_THREADS);
    if (session == NULL) {
        fprintf(stderr, "Error: Failed to create session ONNX Runtime.\n");
        g_ort->ReleaseEnv(env);
        return -1;
    }
    printf("DONE: create_ort_session;\n\n");
    
    /////////////////////////////////////////////////////////
    //////////////////// INFERENCE START ////////////////////
    // Start a parallel region using OpenMP to enable multi-threading
    #pragma omp parallel
    {   
        // Use dynamic scheduling to distribute batches across threads
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
            // Calculate the size of the current batch; handle cases where the last batch is smaller than BATCH_SIZE
            size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;
            
            #pragma omp critical
            {
                printf("Thread %d processing batch %zu to %zu\n", omp_get_thread_num(), i, i + current_batch_size - 1);
            }

            ///////////// Prepare inputs for batch /////////////
            // Select the subset of texts and labels for the current batch
            char** batch_texts = &texts[i];
            char*** batch_labels = (same_labels) ? labels : &labels[i]; // Choose labels based on whether they are the same for all texts
            size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

            // Prepare the inputs for the current batch, which includes formatting the text and labels
            char** prepared_inputs = prepare_inputs(batch_texts, batch_labels, current_batch_size, batch_num_labels, same_labels, prompt_first);
            
            ///////////// Tokenize /////////////
            // Tokenize the inputs using the tokenizer handler, ensuring the tokenized texts fit within MAX_LENGTH
            TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, current_batch_size, MAX_LENGTH);

            ///////////// ONNX /////////////
            OrtValue* input_ids_tensor = NULL;
            OrtValue* attention_mask_tensor = NULL;    

            // Prepare input tensors for the ONNX model; if this fails, log the error and continue to the next batch
            if (prepare_input_tensors(&tokenized, &input_ids_tensor,  &attention_mask_tensor) != 0) {
                #pragma omp critical
                {
                    fprintf(stderr, "Error: Failed to prepare input tensors for batch %zu to %zu\n", i, i + current_batch_size - 1);
                }
                // Free mem and continue with next batch
                free_prepared_inputs(prepared_inputs, current_batch_size);
                free_tokenized_inputs(&tokenized);
                continue;
            }

            ///////////// Run inference /////////////
            OrtValue* output_tensor = run_inference(session, input_ids_tensor, attention_mask_tensor);
            if (output_tensor == NULL) {
                #pragma omp critical
                {
                    fprintf(stderr, "Model inference error for batch %zu to %zu\n", i, i + current_batch_size - 1);
                }
                // Free mem and continue with next batch
                g_ort->ReleaseValue(input_ids_tensor);
                g_ort->ReleaseValue(attention_mask_tensor);
                free_prepared_inputs(prepared_inputs, current_batch_size);
                free_tokenized_inputs(&tokenized);
                continue;
            }

            ///////////// Decoding /////////////
            // Process the output tensor and decode the classification results for the current batch
            process_output_tensor(output_tensor, g_ort, same_labels, batch_labels, batch_num_labels, num_labels_size, THRESHOLD,
                                current_batch_size,batch_texts ,classification_type);

            ///////////// Free batch resources /////////////
            // Release memory used by the prepared inputs, tokenized data, and tensors for the current batch
            free_prepared_inputs(prepared_inputs, current_batch_size);
            free_tokenized_inputs(&tokenized);
            g_ort->ReleaseValue(input_ids_tensor);
            g_ort->ReleaseValue(attention_mask_tensor);
            g_ort->ReleaseValue(output_tensor);
        }
    }
    // Free tokenizer 
    tokenizers_free(tokenizer_handler);

    return 0;
}
