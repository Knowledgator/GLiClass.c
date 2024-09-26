#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tokenizers_c.h"
#include <stdbool.h>

/**
 * Structure to store tokenized data for a batch of inputs.
 * 
 * Contains input IDs, token type IDs, and attention masks for each tokenized input.
 * Also includes the batch size (number of texts) and the sequence length (max tokens per text).
 */
typedef struct {
    int** input_ids;        /**< Array of token IDs for each input text. */
    int** token_type_ids;   /**< Array of token type IDs for each input text. */
    int** attention_mask;   /**< Array indicating which tokens are actual tokens (1) and which are padding (0). */
    size_t batch_size;      /**< Number of input texts in the batch. */
    size_t seq_length;       /**< Maximum sequence length for the input texts. */
} TokenizedInputs;

TokenizedInputs tokenize_inputs(TokenizerHandle tokenizer, const char* inputs[], size_t num_texts, size_t max_length);
void print_tokenized_inputs(const TokenizedInputs* tokenized);
void free_tokenized_inputs(TokenizedInputs* tokenized);
TokenizerHandle create_tokenizer(const char* filepath);

#endif // TOKENIZER_H
