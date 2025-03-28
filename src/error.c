#include "error.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

GLiClassStatus* set_error(GLiClassStatusCode code, const char* format, ...) {
    va_list args;
    va_start(args, format);

    GLiClassStatus* status = (GLiClassStatus*)calloc(1, sizeof(GLiClassStatus*));
    if (!status) {
        fprintf(stderr, "Unable to allocate status");
        exit(1);
    }
    status->code = code;
    vsnprintf(status->msg, GLICLASS_STATUS_MESSAGE_SIZE, format, args);
    va_end(args);
    return status;
}