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
char* read_file(const char* filename);

/**
 * Parses a JSON string to extract model configs.
 *
 * @param json_string The JSON string to parse.
 * @return Model configs.
 * IMPORTANT: The caller is responsible for releasing the model configs.
 */
GLiClassModelConfig* parse_model_config_json(const char* json_string); 
#endif // READ_DATA_H