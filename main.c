#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"
#include "onnxruntime_c_api.h"

// Project includes (folder include)
#include "postprocessor.h"
#include "model.h"
#include "tokenizer.h" 
#include "preprocessor.h"
#include "read_data.h"

// Ini variables for data
char** texts = NULL;
size_t num_texts = 0;
// char** labels = NULL;
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

    ///////////// Prepare inputs /////////////
    parse_json(json_string, &texts, &num_texts, &labels, &num_labels, &num_labels_size, &same_labels, &classification_type);
    bool prompt_first = false;
    printf("DONE: parse_json;\n");
    printf("%s\n\n", classification_type);

    char** prepared_inputs = prepare_inputs(texts, labels, num_texts, num_labels, same_labels, prompt_first);
    printf("DONE: prepared_inputs;\n");  
    

    ///////////// Tokenize /////////////
    TokenizerHandle tokenizer_handler = create_tokenizer(tokenizer_path);
    if (!tokenizer_handler) {
        return 1; // This error is created in create_tokenizer
    }
    printf("DONE: create_tokenizer;\n");  

    TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, num_texts, max_length);
    printf("DONE: tokenize_inputs;\n");  


    ///////////// ONNX /////////////
    initialize_ort_api();
    printf("DONE: initialize_ort_api;\n");

    OrtEnv* env = initialize_ort_environment();
    if (env == NULL) {
        fprintf(stderr, "Error: Failed to initialize ONNX Runtime.\n");
        return -1;
    }
    printf("DONE: initialize_ort_environment;\n");

    OrtSession* session = create_ort_session(env, model_path);
    if (session == NULL) {
        fprintf(stderr, "Error: Failed to create session ONNX Runtime.\n");
        g_ort->ReleaseEnv(env);
        return -1;
    }
    printf("DONE: create_ort_session;\n");

    OrtValue* input_ids_tensor = NULL;
    OrtValue* attention_mask_tensor = NULL;    

    if (prepare_input_tensors(&tokenized, &input_ids_tensor,  &attention_mask_tensor) != 0) {
        fprintf(stderr, "Error: Failed to prepare input tensors.\n");
        // Free mem
        g_ort->ReleaseSession(session);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    printf("DONE: prepare_input_tensors\n");


    ///////////// Run inference /////////////
    OrtValue* output_tensor = run_inference(session, input_ids_tensor, attention_mask_tensor);
    if (output_tensor == NULL) {
        fprintf(stderr, "Model inference error.\n");
        // Free mem
        g_ort->ReleaseValue(input_ids_tensor);
        g_ort->ReleaseValue(attention_mask_tensor);
        g_ort->ReleaseSession(session);
        g_ort->ReleaseEnv(env);
        return -1;
    }
    printf("DONE: run_inference\n\n");


    ///////////// Decoding /////////////
    process_output_tensor(output_tensor, g_ort, same_labels, labels, num_labels, num_labels_size, threshold, num_texts);


    ///////////// Free mem /////////////
    free_prepared_inputs(prepared_inputs, num_texts);
    free_tokenized_inputs(&tokenized);
    tokenizers_free(tokenizer_handler);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseEnv(env);

    return 0;
}
