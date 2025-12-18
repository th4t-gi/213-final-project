#pragma once

#include "utils.h"

void compute_batch(
    global_params_t* params,
    global_params_t* gpu_params,
    const size_t batch_size,
    const size_t k,
    code_t* branches_arr,
    degree_t* degrees_arr,
    float* gpu_partition_values);

__global__ void kernel(
    global_params_t* params,
    const size_t batch_size,
    const size_t k,
    code_t* branches_arr,
    degree_t* degrees_arr,
    float* partition_values);