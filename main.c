#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"
#include "onnxruntime_c_api.h"
#include <omp.h>

#include "time.h"

// Project includes (folder include)
#include "postprocessor.h"
#include "model.h"
#include "tokenizer.h" 
#include "preprocessor.h"
#include "read_data.h"

// Ini variables for data
char** texts = NULL;
size_t num_texts = 0;
char*** labels = NULL;
size_t* num_labels = NULL;
size_t num_labels_size = 0;
bool same_labels = false;
char* classification_type = NULL;

const char* tokenizer_path = "tokenizer/tokenizer.json";
const char* model_path = "onnx/model.onnx";
size_t max_length = 1024; 
float threshold = 0.5f;

const OrtApi* g_ort = NULL;

#define BATCH_SIZE 2

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s /path/to/your_data.json\n", argv[0]);
        printf("NOTE: use this option only if you sure that all required model parts are initialized correctly\n\n");
        printf("Recomended option\n");
        printf("Usage: ./run_GLiClass.sh knowledgator/gliclass-small-v1.0 /path/to/your_data.json\n");
        return 1;
    }
    // reading data from json file
    char* json_string = read_file(argv[1]);
    if (!json_string) {
        return 1;
    }

    // clock_t start, end;
    // double cpu_time_used;
    double start_time, end_time, cpu_time_used;

    ///////////// Prepare inputs /////////////
    parse_json(json_string, &texts, &num_texts, &labels, &num_labels, &num_labels_size, &same_labels, &classification_type);
    bool prompt_first = false;
    printf("DONE: parse_json;\n");
    if (classification_type == NULL){
        printf("classification type is not provided\n");
        return 1;
    }
    ///////////// intializing part /////////////
    initialize_ort_api();
    printf("DONE: initialize_ort_api;\n");

    OrtEnv* env = initialize_ort_environment();
    if (env == NULL) {
        fprintf(stderr, "Error: Failed to initialize ONNX Runtime.\n");
        return -1;
    }
    printf("DONE: initialize_ort_environment;\n");

    OrtSession* session = create_ort_session(env, model_path, 8);
    if (session == NULL) {
        fprintf(stderr, "Error: Failed to create session ONNX Runtime.\n");
        g_ort->ReleaseEnv(env);
        return -1;
    }
    printf("DONE: create_ort_session;\n");

    TokenizerHandle tokenizer_handler = create_tokenizer(tokenizer_path);
    if (!tokenizer_handler) {
        return 1; // This error is created in create_tokenizer
    }
    printf("DONE: create_tokenizer;\n");  
    
    ////////////////////////////////////////////////
    //////////////////// INFERENCE START ////////////
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_texts; i += BATCH_SIZE) {
            size_t current_batch_size = (i + BATCH_SIZE > num_texts) ? (num_texts - i) : BATCH_SIZE;
            
            #pragma omp critical
            {
                printf("Thread %d processing batch %zu to %zu\n", omp_get_thread_num(), i, i + current_batch_size - 1);
            }

            ///////////// Prepare inputs for batch /////////////
            char** batch_texts = &texts[i];
            char*** batch_labels = (same_labels) ? labels : &labels[i];
            size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i];

            char** prepared_inputs = prepare_inputs(batch_texts, batch_labels, current_batch_size, batch_num_labels, same_labels, prompt_first);
            
            #pragma omp critical
            {
                printf("DONE: prepared_inputs for batch %zu to %zu\n", i, i + current_batch_size - 1);
            }

            ///////////// Tokenize /////////////
            TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, current_batch_size, max_length);
            
            #pragma omp critical
            {
                printf("DONE: tokenize_inputs for batch %zu to %zu\n", i, i + current_batch_size - 1);
            }

            ///////////// ONNX /////////////
            OrtValue* input_ids_tensor = NULL;
            OrtValue* attention_mask_tensor = NULL;    

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

            #pragma omp critical
            {
                printf("DONE: prepare_input_tensors for batch %zu to %zu\n", i, i + current_batch_size - 1);
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

            #pragma omp critical
            {
                printf("DONE: run_inference for batch %zu to %zu\n", i, i + current_batch_size - 1);
            }

            ///////////// Decoding /////////////
            process_output_tensor(output_tensor, g_ort, same_labels, batch_labels, batch_num_labels, num_labels_size, threshold, current_batch_size,batch_texts ,classification_type);

            ///////////// Free batch resources /////////////
            free_prepared_inputs(prepared_inputs, current_batch_size);
            free_tokenized_inputs(&tokenized);
            g_ort->ReleaseValue(input_ids_tensor);
            g_ort->ReleaseValue(attention_mask_tensor);
            g_ort->ReleaseValue(output_tensor);
        }
    }
    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;
    printf("Время выполнения: %f секунд\n", cpu_time_used);

    tokenizers_free(tokenizer_handler);

    return 0;
}
