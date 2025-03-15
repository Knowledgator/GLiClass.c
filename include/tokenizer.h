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
    int64_t** input_ids;        /**< Array of token IDs for each input text. */
    int64_t** token_type_ids;   /**< Array of token type IDs for each input text. */
    int64_t** attention_mask;   /**< Array indicating which tokens are actual tokens (1) and which are padding (0). */
    size_t batch_size;      /**< Number of input texts in the batch. */
    size_t seq_length;       /**< Maximum sequence length for the input texts. */
} TokenizedInputs;

/**
 * Structure to store tokenized data for a single input.
 * 
 * Contains input IDs, token type IDs, and attention masks for the input.
 * Also includes the sequence length (max tokens per text).
 */
typedef struct {
    int64_t* input_ids;        /**< Array of token IDs for each input text. */
    int64_t* token_type_ids;   /**< Array of token type IDs for each input text. */
    int64_t* attention_mask;   /**< Array indicating which tokens are actual tokens (1) and which are padding (0). */
    size_t seq_length;       /**< Maximum sequence length for the input texts. */
} TokenizedInput;

/**
 * Tokenizes a batch of input texts using the provided tokenizer.
 *
 * @param tokenizer The tokenizer handle to use for tokenization.
 * @param inputs An array of input texts to be tokenized.
 * @param num_texts The number of input texts in the batch.
 * @param max_length The maximum length of tokens for each text. Sequences longer than this will be truncated.
 * @return A TokenizedInputs structure containing token IDs, token type IDs, and attention masks for the input texts.
 *         The caller is responsible for freeing the memory allocated for the returned structure.
 */
TokenizedInputs tokenize_inputs(
    TokenizerHandle tokenizer, 
    const char* inputs[], 
    const size_t num_texts, 
    const size_t max_length
);

/**
 * Tokenizes a batch of input texts using the provided tokenizer.
 *
 * @param tokenizer The tokenizer handle to use for tokenization.
 * @param inputs An array of input texts to be tokenized.
 * @param num_texts The number of input texts in the batch.
 * @param max_length The maximum length of tokens for each text. Sequences longer than this will be truncated.
 * @return A TokenizedInputs structure containing token IDs, token type IDs, and attention masks for the input texts.
 *         The caller is responsible for freeing the memory allocated for the returned structure.
 */
TokenizedInput tokenize_input(
    TokenizerHandle tokenizer, 
    const char* input, 
    const size_t max_length
);


/**
 * Prints the tokenized inputs including input IDs, token type IDs, and attention masks for each input text.
 *
 * @param tokenized Pointer to the TokenizedInputs structure to be printed.
 */
void print_tokenized_inputs(const TokenizedInputs* tokenized);

/**
 * Frees the memory allocated for the tokenized inputs including input IDs, token type IDs, and attention masks.
 *
 * @param tokenized Pointer to the TokenizedInputs structure to be freed.
 */
void free_tokenized_inputs(TokenizedInputs* tokenized);

/**
 * Frees the memory allocated for the tokenized inputs including input IDs, token type IDs, and attention masks.
 *
 * @param tokenized Pointer to the TokenizedInput structure to be freed.
 */
void free_tokenized_input(TokenizedInput* tokenized);

/**
 * Creates a tokenizer handle from a JSON configuration file.
 *
 * @param filepath The path to the JSON file containing tokenizer settings.
 * @return A TokenizerHandle initialized with the tokenizer settings from the file, or NULL if the file could not be read.
 *         The caller is responsible for freeing the tokenizer handle after use.
 */
TokenizerHandle create_tokenizer(const char* filepath);

#endif // TOKENIZER_H
