#include "read_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "cJSON.h"

#include "error.h"

GLiClassStatus read_file(const char* filename, char** content) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        set_error("Error: Failed to open file %s\n", filename);
        return GC_LOGICAL_ERROR;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    *content = (char*)calloc(length + 1, sizeof(char));
    fread(content, 1, length, file);
    content[length] = '\0';
    fclose(file);
    return GC_OK;
}


GLiClassStatus parse_model_config_json(const char* json_string, GLiClassModelConfig** config_out) {
    GLiClassModelConfig* config = (GLiClassModelConfig*)calloc(1, sizeof(GLiClassModelConfig));
    if (!config) {
        fprintf(stderr, "Unable to allocate model config");
        return GC_MEMORY_ERROR;
    }

    // Parse json
    cJSON* json = cJSON_Parse(json_string);
    if (!json) {
        free(config);
        set_error("Failed to parse JSON: %s\n", cJSON_GetErrorPtr());
        return GC_FILE_ERROR;
    }
    
    // Get array texts
    cJSON* prompt_first_field = cJSON_GetObjectItemCaseSensitive(json, "prompt_first");
    if (prompt_first_field && cJSON_IsBool(prompt_first_field)) {
        config->prompt_first = cJSON_IsTrue(prompt_first_field);
    } else {
        free(config);
        cJSON_Delete(json);
        set_error("Unexpected config format, expected 'prompt_first' field of bool type");
        return GC_FILE_ERROR;
    }
    
    cJSON_Delete(json);  // free memory
    *config_out = config;
    return GC_OK;
}