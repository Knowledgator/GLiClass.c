#ifndef READ_DATA_H
#define READ_DATA_H

#include <stddef.h>
#include <stdbool.h>
#include "GLiClass/gliclass_common.h"

/**
 * Reads the entire content of a file and returns it as a string.
 *
 * @param filename The name of the file to read.
 * @return A dynamically allocated string containing the file content, or NULL if the file could not be opened.
 *         The caller is responsible for freeing the allocated memory.
 */
GLiClassStatus* read_file(const char* filename, char** content);

/**
 * Parses a JSON string to extract model configs.
 *
 * @param json_string The JSON string to parse.
 * @return Model configs.
 * IMPORTANT: The caller is responsible for releasing the model configs.
 */
GLiClassStatus* parse_model_config_json(const char* json_string, GLiClassModelConfig** config_out); 
#endif // READ_DATA_H