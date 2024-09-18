#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#include "tokenizer.h"

TokenizedInputs tokenize_inputs(TokenizerHandle tokenizer, const char* inputs[], size_t num_texts) {
    TokenizerEncodeResult* results = (TokenizerEncodeResult*)malloc(num_texts * sizeof(TokenizerEncodeResult));
    if (!results) {
        fprintf(stderr, "Error while allocating memmory for tokenization results\n");
        exit(1);
    }

    // Get len of each text
    size_t* input_lengths = (size_t*)malloc(num_texts * sizeof(size_t));
    for (size_t i = 0; i < num_texts; ++i) {
        input_lengths[i] = strlen(inputs[i]);
    }

    tokenizers_encode_batch(tokenizer, inputs, input_lengths, num_texts, 1, results);

    // Calculate max sequence lenght
    size_t max_len = 0;
    for (size_t i = 0; i < num_texts; ++i) {
        if (results[i].len > max_len) {
            max_len = results[i].len;
        }
    }

    // Mem alloc for tokenized data
    TokenizedInputs tokenized;
    tokenized.input_ids = (int**)malloc(num_texts * sizeof(int*));
    tokenized.token_type_ids = (int**)malloc(num_texts * sizeof(int*));
    tokenized.attention_mask = (int**)malloc(num_texts * sizeof(int*));
    tokenized.batch_size = num_texts;
    tokenized.seq_length = max_len;

    for (size_t i = 0; i < num_texts; ++i) {
        tokenized.input_ids[i] = (int*)malloc(max_len * sizeof(int));
        tokenized.token_type_ids[i] = (int*)malloc(max_len * sizeof(int));
        tokenized.attention_mask[i] = (int*)malloc(max_len * sizeof(int));

        for (size_t j = 0; j < max_len; ++j) {
            if (j < results[i].len) {
                tokenized.input_ids[i][j] = results[i].token_ids[j];
                tokenized.token_type_ids[i][j] = 0;  // В данном случае для простоты ставим 0
                tokenized.attention_mask[i][j] = 1;  // 1, если токен присутствует
            } else {
                tokenized.input_ids[i][j] = 0;  // Padding
                tokenized.token_type_ids[i][j] = 0;
                tokenized.attention_mask[i][j] = 0;  // Padding токен не учитывается
            }
        }
    }

    tokenizers_free_encode_results(results, num_texts);
    free(input_lengths);

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

TokenizerHandle create_tokenizer(const char* filepath) {
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
    char* json = (char*)malloc(json_len + 1);
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
