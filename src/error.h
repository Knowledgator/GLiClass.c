#ifndef ERROR_H
#define ERROR_H

#define GLICLASS_STATUS_MESSAGE_SIZE 256

#include <stddef.h>
#include "GLiClass/gliclass_common.h"

GLiClassStatus* set_error(GLiClassStatusCode code, const char* format, ...);

#endif