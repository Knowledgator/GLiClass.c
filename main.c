#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"
#include "onnxruntime_c_api.h"
#include <omp.h>
#include <mqueue.h>
#include <pthread.h>


// Project includes (folder include)
#include "preprocessor.h"
#include "read_data.h"
#include "postprocessor.h"
#include "model.h"
#include "tokenizer.h" 
#include "preprocessor.h"
#include "read_data.h"
#include "paths.h"
#include "configs.h"
#include "parallel_processor.h"

// Ini variables for data
char** texts = NULL;                // Array of strings containing texts to classify
size_t num_texts = 0;               // Number of texts in the 'texts' array
char*** labels = NULL;              // An array of labels for each text; there can be multiple labels for each text
size_t* num_labels = NULL;          // Array containing the number of tags for each text
size_t num_labels_size = 0;         // Total size of the array of labels 
bool same_labels = false;           // Flag indicating whether the same labels are used for all texts
char* classification_type = NULL;   // Classification type (e.g. single-label, multi-label)

const OrtApi* g_ort = NULL;         // Global pointer to ONNX Runtime API for performing model inference

// Mutex declarations
pthread_mutex_t queue_mutex;

// Buffers for input and output
OrtValue** input_ids_tensors = NULL;
OrtValue** attention_mask_tensors = NULL;
OrtValue** output_tensors = NULL;

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
    ///////////// Prepare inputs /////////////
    parse_json(json_string, &texts, &num_texts, &labels, &num_labels, &num_labels_size, &same_labels, &classification_type);
    printf("DONE: parse_json;\n");
    if (classification_type == NULL){
        printf("classification type is not provided\n");
        return 1;
    }
    free(json_string);
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
    // Initialize queue mutex
    pthread_mutex_init(&queue_mutex, NULL);

    // Allocate memory for tensors
    size_t num_batches = (num_texts + BATCH_SIZE - 1) / BATCH_SIZE;
    input_ids_tensors = malloc(sizeof(OrtValue*) * num_batches);
    attention_mask_tensors = malloc(sizeof(OrtValue*) * num_batches);
    output_tensors = malloc(sizeof(OrtValue*) * num_batches);

    double start_time, end_time;
    start_time = omp_get_wtime();

    // Parallel preprocessing
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
        prepare_input_tensors(&tokenized, &input_ids_tensors[i / BATCH_SIZE], &attention_mask_tensors[i / BATCH_SIZE]);

        // Clean up memory
        free_prepared_inputs((char**)prepared_inputs, current_batch_size);
        free_tokenized_inputs(&tokenized);
    }

    // Inference stage - processing batches
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        #ifdef USE_CUDA // GPU
        pthread_mutex_lock(&queue_mutex);
        output_tensors[i] = run_inference(session, input_ids_tensors[i], attention_mask_tensors[i]);
        pthread_mutex_unlock(&queue_mutex);
        #else
        output_tensors[i] = run_inference(session, input_ids_tensors[i], attention_mask_tensors[i]);
        #endif
    }

    // Postprocess stage - processing batches
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_batches; i++) {
        size_t current_batch_size = (i == num_batches - 1) ? (num_texts - i * BATCH_SIZE) : BATCH_SIZE;

        const char** batch_texts = (const char**)&texts[i * BATCH_SIZE];
        const char*** batch_labels = (const char***)(same_labels ? (void*)labels : (void*)&labels[i * BATCH_SIZE]);
        size_t* batch_num_labels = (same_labels) ? num_labels : &num_labels[i * BATCH_SIZE];

        process_output_tensor(output_tensors[i], g_ort, same_labels, batch_labels, batch_num_labels, num_labels_size, THRESHOLD,
                              current_batch_size, batch_texts, classification_type);
        // Free output tensor after processing
        g_ort->ReleaseValue(output_tensors[i]);
    }
    
    end_time = omp_get_wtime();
    printf("Execution time: %f seconds\n", end_time - start_time);
    // Free resources
    for (size_t i = 0; i < num_batches; i++) {
        g_ort->ReleaseValue(input_ids_tensors[i]);
        g_ort->ReleaseValue(attention_mask_tensors[i]);
        // g_ort->ReleaseValue(output_tensors[i]);
    }

    free(input_ids_tensors);
    free(attention_mask_tensors);
    free(output_tensors);

    // Free tokenizer
    tokenizers_free(tokenizer_handler);
    // Free onnx
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    pthread_mutex_destroy(&queue_mutex);
    return 0;
}