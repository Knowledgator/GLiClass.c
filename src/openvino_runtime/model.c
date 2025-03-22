#include "model.h"

#include <stdio.h>
#include "../utils.h"

int prepare_input_tensor_openvino(
    TokenizedInput* tokenized, ov_tensor_t** input_ids_tensor, ov_tensor_t** attention_mask_tensor
) {
    ov_shape_t input_shape = {0};
    int64_t dims[] = {1, tokenized->seq_length};
    ov_shape_create(2, dims, &input_shape);
    ov_element_type_e input_type = I64;
    ov_status_e status = ov_tensor_create_from_host_ptr(input_type, input_shape, tokenized->input_ids, input_ids_tensor);
    if (status != OK) {
        fprintf(stderr, "Unable to allocate intput_ids_tensor");
        return -1;
    }
    ov_tensor_create_from_host_ptr(input_type, input_shape, tokenized->attention_mask, attention_mask_tensor);
    if (status != OK) {
        fprintf(stderr, "Unable to allocate attention_mask_tensor");
        ov_tensor_free(*input_ids_tensor);
        return -1;
    }
    ov_shape_free(&input_shape);
    return 0;
}


ov_tensor_t* run_inference_openvino(
    ov_compiled_model_t* model, ov_tensor_t* input_ids_tensor, ov_tensor_t* attention_mask_tensor
) {
    ov_infer_request_t* infer_request = NULL;
    ov_status_e status = ov_compiled_model_create_infer_request(model, &infer_request);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to create infer request: %s\n", e);
        return NULL;
    }

    status = ov_infer_request_set_tensor(infer_request, "input_ids", input_ids_tensor);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to add input_ids_tensor: %s\n", e);
        ov_infer_request_free(infer_request);
        return NULL;
    }
    status = ov_infer_request_set_tensor(infer_request, "attention_mask", attention_mask_tensor);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to add attention_mask_tensor: %s\n", e);
        ov_infer_request_free(infer_request);
        return NULL;
    }

    status = ov_infer_request_infer(infer_request);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Inference failed: %s\n", e);
        ov_infer_request_free(infer_request);
        return NULL;
    }

    ov_tensor_t* output_tensor = NULL;
    status = ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor);
    if (status != OK) {
        const char* e = ov_get_error_info(status);
        fprintf(stderr, "Unable to get results: %s\n", e);
        ov_infer_request_free(infer_request);
        return NULL;
    }
    ov_infer_request_free(infer_request);
    return output_tensor;
}