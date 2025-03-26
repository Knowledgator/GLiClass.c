#ifndef ERROR_H
#define ERROR_H

#define GLICLASS_STATUS_MESSAGE_SIZE 256

#include <stddef.h>

typedef struct GLiClassError {
    const char message[GLICLASS_STATUS_MESSAGE_SIZE];
} GLiClassError;

extern const GLiClassError* last_error;

const GLiClassError* last_error = NULL;

void set_error(const char* format, ...) {
    va_list args;
    va_start(args, format);

    if (last_error) {
        free(last_error);
    }
    last_error = (GLiClassError*)calloc(1, sizeof(GLiClassError));
    snprintf(last_error->message, GLICLASS_STATUS_MESSAGE_SIZE, args);
}

#endif