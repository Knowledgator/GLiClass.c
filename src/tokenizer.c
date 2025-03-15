#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#ifndef _WIN32
    #include <unistd.h>
#else
    #include <io.h>

    #define access _access
    #define F_OK 0
#endif

TokenizedInputs tokenize_inputs(TokenizerHandle tokenizer, const char** inputs, size_t num_texts, size_t max_length) {
    TokenizerEncodeResult* results = (TokenizerEncodeResult*)calloc(num_texts, sizeof(TokenizerEncodeResult));
    if (!results) {
        fprintf(stderr, "Error while allocating memmory for tokenization results\n");
        exit(1);
    }

    // Get len of each text
    size_t* input_lengths = (size_t*)calloc(num_texts, sizeof(size_t));
    for (size_t i = 0; i < num_texts; ++i) {
        input_lengths[i] = strlen(inputs[i]);
    }

    int add_special_tokens = 1;
    tokenizers_encode_batch(tokenizer, inputs, input_lengths, num_texts, add_special_tokens, results);

    // We trim the sequences to max_length and find the maximum length after trimming
    size_t* seq_lengths = (size_t*)calloc(num_texts, sizeof(size_t));
    if (!seq_lengths) {
        fprintf(stderr, "Error while allocating memory for sequence lengths\n");
        free(results);
        free(input_lengths);
        exit(1);
    }

    size_t seq_length = 0; // This will be the length of the longest sequence after trimming.
    for (size_t i = 0; i < num_texts; ++i) {
        if (results[i].len > max_length) {
            seq_lengths[i] = max_length;
        } else {
            seq_lengths[i] = results[i].len;
        }
        if (seq_lengths[i] > seq_length) {
            seq_length = seq_lengths[i];
        }
    }

    // Mem alloc for tokenized data
    TokenizedInputs tokenized;
    tokenized.input_ids = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized.token_type_ids = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized.attention_mask = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized.batch_size = num_texts;
    tokenized.seq_length = seq_length;

    for (size_t i = 0; i < num_texts; ++i) {
        tokenized.input_ids[i] = (int64_t*)calloc(seq_length, sizeof(int64_t));
        tokenized.token_type_ids[i] = (int64_t*)calloc(seq_length, sizeof(int64_t));
        tokenized.attention_mask[i] = (int64_t*)calloc(seq_length, sizeof(int64_t));

        for (size_t j = 0; j < seq_length; ++j) {
            if (j < results[i].len) {
                if (j >= max_length) {
                    // If the length exceeds max_length, cut it off
                    break;
                }
                tokenized.input_ids[i][j] = results[i].token_ids[j];
                tokenized.token_type_ids[i][j] = 0;  // In this case, for simplicity, we set it to 0
                tokenized.attention_mask[i][j] = 1;  // 1 if token is exists
            } else {
                tokenized.input_ids[i][j] = 0;  // Padding
                tokenized.token_type_ids[i][j] = 0;
                tokenized.attention_mask[i][j] = 0;  // Padding токен не учитывается
            }
        }
    }

    tokenizers_free_encode_results(results, num_texts);
    free(input_lengths);
    free(seq_lengths);

    return tokenized;
}

TokenizedInput tokenize_input(TokenizerHandle tokenizer, const char* input, size_t max_length) {
    TokenizerEncodeResult result;
    size_t input_length = strlen(input);

    int add_special_tokens = 1;
    tokenizers_encode(tokenizer, input, input_length, add_special_tokens, &result);

    size_t seq_length = 0; // This will be the length of the longest sequence after trimming.
    if (result.len > max_length) {
        seq_length = max_length;
    } else {
        seq_length = result.len;
    }

    // Mem alloc for tokenized data
    TokenizedInput tokenized;
    tokenized.input_ids = (int64_t*)calloc(seq_length, sizeof(int64_t));
    tokenized.token_type_ids = (int64_t*)calloc(seq_length, sizeof(int64_t));
    tokenized.attention_mask = (int64_t*)calloc(seq_length, sizeof(int64_t));
    tokenized.seq_length = seq_length;

    for (size_t j = 0; j < seq_length; ++j) {
        if (j < result.len) {
            if (j >= max_length) {
                // If the length exceeds max_length, cut it off
                break;
            }
            tokenized.input_ids[j] = result.token_ids[j];
            tokenized.token_type_ids[j] = 0;  // In this case, for simplicity, we set it to 0
            tokenized.attention_mask[j] = 1;  // 1 if token is exists
        } else {
            tokenized.input_ids[j] = 0;  // Padding
            tokenized.token_type_ids[j] = 0;
            tokenized.attention_mask[j] = 0;  // Padding токен не учитывается
        }
    }

    tokenizers_free_encode_results(&result, 1);
    return tokenized;
}

void print_tokenized_inputs(const TokenizedInputs* tokenized) {
    for (size_t i = 0; i < tokenized->batch_size; ++i) {
        printf("Input %zu:\n", i);
        printf("input_ids: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%d, ", tokenized->input_ids[i][j]);
        }
        printf("]\n");

        printf("token_type_ids: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%d, ", tokenized->token_type_ids[i][j]);
        }
        printf("]\n");

        printf("attention_mask: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%d, ", tokenized->attention_mask[i][j]);
        }
        printf("]\n");        
    }
}

void free_tokenized_inputs(TokenizedInputs* tokenized) {
    for (size_t i = 0; i < tokenized->batch_size; ++i) {
        free(tokenized->input_ids[i]);
        free(tokenized->token_type_ids[i]);
        free(tokenized->attention_mask[i]);
    }
    free(tokenized->input_ids);
    free(tokenized->token_type_ids);
    free(tokenized->attention_mask);
}

void free_tokenized_input(TokenizedInput* tokenized) {
    free(tokenized->input_ids);
    free(tokenized->token_type_ids);
    free(tokenized->attention_mask);
}

TokenizerHandle create_tokenizer(const char* filepath) {
    // Check existence
    if (access(filepath, F_OK) != 0) {
        fprintf(stderr, "Error: Tokenizer file not found at path: %s\n", filepath);
        return NULL;
    }

    // Read tokenizer.json
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Cant open file %s\n", filepath);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t json_len = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for JSON
    char* json = (char*)calloc(json_len + 1, sizeof(char));
    if (!json) {
        fprintf(stderr, "Cant allocate memory for JSON\n");
        fclose(file);
        return NULL;
    }

    // Read file
    size_t read_len = fread(json, 1, json_len, file);
    fclose(file);
    if (read_len != json_len) {
        fprintf(stderr, "Failed to read %s\n", filepath);
        free(json);
        return NULL;
    }
    json[json_len] = '\0'; // Add last null sym

    // Initialize tokenizer
    TokenizerHandle handle = tokenizers_new_from_str(json, json_len);
    free(json); // Free memory after initializing

    if (!handle) {
        fprintf(stderr, "Cant create tokenizer from %s\n", filepath);
        return NULL;
    }

    return handle;
}
