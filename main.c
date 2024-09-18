#include <stdio.h>
#include <stdlib.h>
#include "tokenizers_c.h"

int main() {
    // Read tokenizer.json
    FILE* file = fopen("tokenizer/tokenizer.json", "rb");
    if (!file) {
        fprintf(stderr, "Cant open file  tokenizer/tokenizer.json\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    size_t json_len = ftell(file);
    fseek(file, 0, SEEK_SET);

    // Allocate memory for  JSON
    char* json = (char*)malloc(json_len + 1);
    if (!json) {
        fprintf(stderr, "Cant allocate memory for JSON\n");
        fclose(file);
        return 1;
    }

    // Read file
    size_t read_len = fread(json, 1, json_len, file);
    fclose(file);
    if (read_len != json_len) {
        fprintf(stderr, "Faild to read tokenizer/tokenizer.json\n");
        free(json);
        return 1;
    }
    json[json_len] = '\0'; // Add las null sym

    // Ini tokenizer
    TokenizerHandle handle = tokenizers_new_from_str(json, json_len);
    free(json); // free mem

    if (!handle) {
        fprintf(stderr, "Cant create tokenizer\n");
        return 1;
    }

    // Tokenize 
    const char* sentence = "Hello world!";
    TokenizerEncodeResult result;
    tokenizers_encode(handle, sentence, strlen(sentence), 1, &result);

    // print
    for (size_t i = 0; i < result.len; ++i) {
        printf("Token ID: %d\n", result.token_ids[i]);
    }

    // Free mem
    tokenizers_free_encode_results(&result, 1);
    tokenizers_free(handle);

    return 0;
}
