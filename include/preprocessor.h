#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <stddef.h>
#include <stdbool.h>

char** prepare_inputs(const char* texts[], const char* labels[], size_t num_texts,
                    size_t num_labels[], bool same_labels, bool prompt_first);
                    
char* prepare_input(const char* text, const char* labels[], size_t num_labels, bool prompt_first);

#endif // PREPROCESSOR_H
