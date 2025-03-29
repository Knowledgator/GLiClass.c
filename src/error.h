#ifndef ERROR_H
#define ERROR_H

#include <stddef.h>
#include "GLiClass/gliclass_common.h"

GLiClassStatus* set_error(GLiClassStatusCode code, const char* format, ...);

#endif