#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tokenizers_c.h"
#include <stdbool.h>

// Structure to store tokenized data
typedef struct {
    int** input_ids;
    int** token_type_ids;
    int** attention_mask;
    size_t batch_size;
    size_t seq_length;  
} TokenizedInputs;

TokenizedInputs tokenize_inputs(TokenizerHandle tokenizer, const char* inputs[], size_t num_texts);
void print_tokenized_inputs(const TokenizedInputs* tokenized);
void free_tokenized_inputs(TokenizedInputs* tokenized);
TokenizerHandle create_tokenizer(const char* filepath);

#endif // TOKENIZER_H
