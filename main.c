#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tokenizers_c.h"

// Project includes (folder include)
#include "tokenizer.h" 
#include "preprocessor.h"

int main() {
    const char* texts[] = {
        "ONNX is an open-source format designed to enable the interoperability of AI models.",
        "Why are you running?",
        "Hello"
    };
    const char* labels[] = {"format", "model", "tool", "cat"};

    size_t num_texts = 3;
    size_t num_labels[] = {4, 4, 4};

    bool same_labels = true;
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
