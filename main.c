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
#include "utils.h"

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
    print_done("parse json;");
    if (classification_type == NULL){
        print_error("classification type is not provided");
        return 1;
    }
    free(json_string);
    ///////////// intializing part /////////////
    TokenizerHandle tokenizer_handler = create_tokenizer(TOKENIZER_PATH);
    if (!tokenizer_handler) {
        return 1; // This error is created in create_tokenizer
    }
    print_done("create tokenizer;");

    initialize_ort_api();
    print_done("initialize ort api;");

    OrtEnv* env = initialize_ort_environment();
    if (env == NULL) {
        print_error("Failed to initialize ONNX Runtime.");
        return -1;
    }
    print_done("initialize ort environment;");

    OrtSession* session = create_ort_session(env, MODEL_PATH, NUM_THREADS);
    if (session == NULL) {
        print_error("Failed to create session ONNX Runtime.");
        g_ort->ReleaseEnv(env);
        return -1;
    }
    print_done("create ort session;");
    
    /////////////////////////////////////////////////////////
    //////////////////// INFERENCE START ////////////////////
    // Initialize queue mutex
    pthread_mutex_init(&queue_mutex, NULL);

    // Allocate memory for tensors
    size_t num_batches = (num_texts + BATCH_SIZE - 1) / BATCH_SIZE;
    input_ids_tensors = malloc(sizeof(OrtValue*) * num_batches);
    attention_mask_tensors = malloc(sizeof(OrtValue*) * num_batches);
    output_tensors = malloc(sizeof(OrtValue*) * num_batches);

    // Parallel preprocessing
    parallel_preprocess(texts, labels, num_labels, num_texts,
                       same_labels, prompt_first, tokenizer_handler,
                       input_ids_tensors, attention_mask_tensors);    

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
    parallel_postprocess(output_tensors, num_batches, num_texts,
                        texts, labels, num_labels,
                        same_labels, num_labels_size, classification_type);
    
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