#include "model.h"

#include <stdio.h>
#include "../utils.h"
#include "../error.h"

GLiClassStatus openvino_create_tensor(
    int64_t* data, ov_element_type_e input_type, ov_shape_t input_shape, ov_tensor_t** tensor
) {
    ov_status_e status = ov_tensor_create_from_host_ptr(
        input_type, input_shape, data, tensor
    );
    if (status != OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to create tensor: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        return GC_INFERENCE_ERROR;
    }
    return GC_OK;
}


GLiClassStatus openvino_prepare_input_tensors(
    const TokenizedInput* tokenized, ov_tensor_t** input_ids_tensor, ov_tensor_t** attention_mask_tensor
) {
    ov_shape_t input_shape = {0};
    int64_t dims[] = {1, tokenized->seq_length};
    ov_status_e ov_status = ov_shape_create(2, dims, &input_shape);
    if (ov_status != OK) {
        const char* e_i = ov_get_error_info(ov_status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to create tensor shape: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        return GC_INFERENCE_ERROR;
    }
    ov_element_type_e input_type = I64;

    GLiClassStatus status = openvino_create_tensor(
        tokenized->input_ids, input_type, input_shape, input_ids_tensor
    );
    if (status != GC_OK) {
        ov_shape_free(&input_shape);
        return status;
    }

    status = openvino_create_tensor(
        tokenized->attention_mask, input_type, input_shape, attention_mask_tensor
    );
    if (status != GC_OK) {
        ov_tensor_free(*input_ids_tensor);
    }

    ov_shape_free(&input_shape);
    return status;
}


GLiClassStatus openvino_run_inference(
    GLiClassOpenVinoSession* session, 
    ov_tensor_t* input_ids_tensor, 
    ov_tensor_t* attention_mask_tensor, 
    ov_tensor_t** output_tensor
) {
    ov_infer_request_t* infer_request = NULL;
    ov_status_e status = ov_compiled_model_create_infer_request(session->model, &infer_request);
    if (status != GC_OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to create infer request: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        return GC_INFERENCE_ERROR;
    }

    status = ov_infer_request_set_tensor(infer_request, "input_ids", input_ids_tensor);
    if (status != GC_OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to add input_ids tensor: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        ov_infer_request_free(infer_request);
        return GC_INFERENCE_ERROR;
    }
    status = ov_infer_request_set_tensor(infer_request, "attention_mask", attention_mask_tensor);
    if (status != GC_OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to add attention_mask tensor: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        ov_infer_request_free(infer_request);
        return GC_INFERENCE_ERROR;
    }

    status = ov_infer_request_infer(infer_request);
    if (status != GC_OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Inference failed: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        ov_infer_request_free(infer_request);
        return GC_INFERENCE_ERROR;
    }

    status = ov_infer_request_get_output_tensor_by_index(infer_request, 0, output_tensor);
    ov_infer_request_free(infer_request);
    if (status != GC_OK) {
        const char* e_i = ov_get_error_info(status);
        const char* e_m = ov_get_last_err_msg();
        set_error("Unable to get results: %s: %s", e_i, e_m);
        ov_free(e_i);
        ov_free(e_m);
        return GC_INFERENCE_ERROR;
    }
    return GC_OK;
}