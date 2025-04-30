#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tokenizers_c.h"
#include <stdbool.h>
#include "GLiClass/gliclass_common.h"

/**
 * Tokenizes a batch of input texts using the provided tokenizer.
 *
 * @param tokenizer The tokenizer handle to use for tokenization.
 * @param inputs An array of input texts to be tokenized.
 * @param num_texts The number of input texts in the batch.
 * @return A TokenizedInputs structure containing token IDs, token type IDs, and attention masks for the input texts.
 *         The caller is responsible for freeing the memory allocated for the returned structure.
 */
GLiClassStatus* tokenize_inputs(
    TokenizerHandle tokenizer, 
    const char** inputs, 
    const size_t num_texts,
    TokenizedInputs* tokenized,
    GLiClassTokensInfo** info
);

/**
 * Tokenizes a batch of input texts using the provided tokenizer.
 *
 * @param tokenizer The tokenizer handle to use for tokenization.
 * @param inputs An array of input texts to be tokenized.
 * @param num_texts The number of input texts in the batch.
 * @return A TokenizedInputs structure containing token IDs, token type IDs, and attention masks for the input texts.
 *         The caller is responsible for freeing the memory allocated for the returned structure.
 */
GLiClassStatus* tokenize_input(
    TokenizerHandle tokenizer, 
    const char* input,
    TokenizedInput* tokenized,
    GLiClassTokensInfo* info
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
GLiClassStatus* create_tokenizer(const char* filepath, TokenizerHandle* tokenizer);

#endif // TOKENIZER_H
