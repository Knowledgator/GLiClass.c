#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"
#include "onnxruntime_c_api.h"
#include <omp.h>

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
    // Start a parallel region using OpenMP to enable multi-threading
    double start_time, end_time;
    start_time = omp_get_wtime();

    int result = process_batches_parallel(
        (const char**)texts, num_texts, labels, num_labels, num_labels_size,
        same_labels, prompt_first, classification_type, tokenizer_handler,
        session, g_ort, THRESHOLD
    );
    
    end_time = omp_get_wtime();
    printf("Execution time: %f seconds\n", end_time - start_time);
    // Free tokenizer 
    tokenizers_free(tokenizer_handler);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}
