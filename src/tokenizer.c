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

#include "error.h"

GLiClassStatus* tokenize_inputs(
    TokenizerHandle tokenizer, 
    const char** inputs, 
    const size_t num_texts,
    const size_t min_length,
    const size_t max_length,
    TokenizedInputs* tokenized,
    GLiClassTokensInfo** info
) {
    TokenizerEncodeResult* results = (
        (TokenizerEncodeResult*)calloc(num_texts, sizeof(TokenizerEncodeResult))
    );
    if (!results) {
        return set_error(GC_MEMORY_ERROR, "Error while allocating memmory for tokenization results");
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
        free(results);
        free(input_lengths);
        return set_error(GC_MEMORY_ERROR, "Error while allocating memory for sequence lengths");
    }

    // Mem alloc for tokenized data
    tokenized->input_ids = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized->token_type_ids = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized->attention_mask = (int64_t**)calloc(num_texts, sizeof(int64_t*));
    tokenized->batch_size = num_texts;
    tokenized->seq_length = 0; // This will be the length of the longest sequence after trimming.
    for (size_t i = 0; i < num_texts; ++i) {
        bool truncated = false;
        if (results[i].len < min_length) {
            seq_lengths[i] = 0;
            truncated = true;

        } else if (results[i].len > max_length) {
            seq_lengths[i] = max_length;
            truncated = true;
        } else {
            seq_lengths[i] = results[i].len;
        }

        if (info && info[i]) {
            info[i]->tokens_num = seq_lengths[i];
            info[i]->truncated = truncated;
        }

        if (seq_lengths[i] > tokenized->seq_length) {
            tokenized->seq_length = seq_lengths[i];
        }
    }

    for (size_t i = 0; i < num_texts; ++i) {
        tokenized->input_ids[i] = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));
        tokenized->token_type_ids[i] = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));
        tokenized->attention_mask[i] = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));

        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            if (j < results[i].len) {
                tokenized->input_ids[i][j] = results[i].token_ids[j];
                // tokenized->token_type_ids[i][j] = 0;  // In this case, for simplicity, we set it to 0 // set by calloc
                tokenized->attention_mask[i][j] = 1;  // 1 if token is exists
            } 
            // else {
            //    tokenized->input_ids[i][j] = 0;  // Padding // set by calloc
            //    tokenized->token_type_ids[i][j] = 0; // set by calloc
            //    tokenized->attention_mask[i][j] = 0; // set by calloc
            // }
        }
    }

    tokenizers_free_encode_results(results, num_texts);
    free(input_lengths);
    free(seq_lengths);
    return NULL;
}

GLiClassStatus* tokenize_input(
    TokenizerHandle tokenizer, 
    const char* input,
    const size_t min_length,
    const size_t max_length,
    TokenizedInput* tokenized,
    GLiClassTokensInfo* info
) {
    TokenizerEncodeResult result;
    size_t input_length = strlen(input);

    int add_special_tokens = 1;
    tokenizers_encode(tokenizer, input, input_length, add_special_tokens, &result);

    bool truncated;
    if (result.len < min_length) {
        tokenized->seq_length = 0;
        if (info) {
            info->truncated = true;
            info->tokens_num = tokenized->seq_length;
        }
        tokenizers_free_encode_results(&result, 1);
        return NULL;
    } else if (result.len > max_length) {
        tokenized->seq_length = max_length;
        truncated = true;
    } else {
        tokenized->seq_length = result.len;
        truncated = false;
    }

    if (info) {
        info->truncated = truncated;
        info->tokens_num = tokenized->seq_length;
    }

    // Mem alloc for tokenized data
    tokenized->input_ids = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));
    tokenized->token_type_ids = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));
    tokenized->attention_mask = (int64_t*)calloc(tokenized->seq_length, sizeof(int64_t));
    if (!tokenized->input_ids || !tokenized->token_type_ids || !tokenized->attention_mask) {
        return set_error(GC_MEMORY_ERROR, "Unable to allocate tokenized inputs");
    }

    for (size_t j = 0; j < tokenized->seq_length; ++j) {
        tokenized->input_ids[j] = result.token_ids[j];
        // tokenized->token_type_ids[j] = 0;  // In this case, for simplicity, we set it to 0
        tokenized->attention_mask[j] = 1;  // 1 if token is exists
    }

    tokenizers_free_encode_results(&result, 1);
    return NULL;
}

void print_tokenized_inputs(const TokenizedInputs* tokenized) {
    for (size_t i = 0; i < tokenized->batch_size; ++i) {
        printf("Input %zu:\n", i);
        printf("input_ids: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%zu, ", tokenized->input_ids[i][j]);
        }
        printf("]\n");

        printf("token_type_ids: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%zu, ", tokenized->token_type_ids[i][j]);
        }
        printf("]\n");

        printf("attention_mask: [");
        for (size_t j = 0; j < tokenized->seq_length; ++j) {
            printf("%zu, ", tokenized->attention_mask[i][j]);
        }
        printf("]\n");        
    }
}

void free_tokenized_inputs(TokenizedInputs* tokenized) {
    if (tokenized->seq_length == 0)
        return;
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
    if (tokenized->seq_length == 0)
        return;
    free(tokenized->input_ids);
    free(tokenized->token_type_ids);
    free(tokenized->attention_mask);
}

GLiClassStatus* create_tokenizer(const char* filepath, TokenizerHandle* tokenizer) {
    // Check existence
    if (access(filepath, F_OK) != 0) {
        return set_error(GC_FILE_ERROR, "Tokenizer file not found at path: %s", filepath);
    }

    // Read tokenizer.json
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        return set_error(GC_FILE_ERROR, "Cant open file %s", filepath);
    }

    fseek(file, 0, SEEK_END);
    size_t json_len = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for JSON
    char* json = (char*)calloc(json_len + 1, sizeof(char));
    if (!json) {
        fclose(file);
        return set_error(GC_MEMORY_ERROR, "Cant allocate memory for JSON");
    }

    // Read file
    size_t read_len = fread(json, 1, json_len, file);
    fclose(file);
    if (read_len != json_len) {
        free(json);
        return set_error(GC_MEMORY_ERROR, "Failed to read %s", filepath);
    }
    json[json_len] = '\0'; // Add last null sym

    // Initialize tokenizer
    *tokenizer = tokenizers_new_from_str(json, json_len);
    free(json); // Free memory after initializing

    if (!(*tokenizer)) {
        return set_error(GC_FILE_ERROR, "Cant create tokenizer from %s", filepath);
    }

    return NULL;
}
