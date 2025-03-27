#ifndef ERROR_H
#define ERROR_H

#define GLICLASS_STATUS_MESSAGE_SIZE 256

#include <stddef.h>

typedef struct GLiClassError {
    const char message[GLICLASS_STATUS_MESSAGE_SIZE];
} GLiClassError;

extern GLiClassError* last_error;

void set_error(const char* format, ...);

#endif