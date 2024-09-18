#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"
#include "cJSON.h"

// Project includes (folder include)
#include "tokenizer.h" 
#include "preprocessor.h"
#include "read_data.h"

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

    char** texts = NULL;
    size_t num_texts = 0;
    char** labels = NULL;
    size_t* num_labels = NULL;
    size_t num_labels_size = 0;
    bool same_labels = false;

    parse_json(json_string, &texts, &num_texts, &labels, &num_labels, &num_labels_size, &same_labels);

    // size_t num_texts = 3;
    // size_t num_labels[] = {4, 4, 4};

    // bool same_labels = true;
    bool prompt_first = false;

    TokenizerHandle tokenizer_handler = create_tokenizer("tokenizer/tokenizer.json");
    if (!tokenizer_handler) {
        return 1; // This error is created in create_tokenizer
    }

    // Prepare inputs
    char** prepared_inputs = prepare_inputs(texts, labels, num_texts, num_labels, same_labels, prompt_first);  
    // Tokenize 
    TokenizedInputs tokenized = tokenize_inputs(tokenizer_handler, prepared_inputs, num_texts);
    // print
    print_tokenized_inputs(&tokenized);

    // Free mem
    free_tokenized_inputs(&tokenized);
    tokenizers_free(tokenizer_handler);

    return 0;
}
