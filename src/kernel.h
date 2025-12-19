#pragma once

#include <algorithm>
#include "utils.h"

void compute_batch(
    global_params_t* params,
    global_params_t* gpu_params,
    const size_t batch_size,
    const size_t k,
    code_t* branches_arr,
    degree_t* degrees_arr,
    float* gpu_output_matrix);

__device__ int falling_factorial(int x, int n);
__device__ double interaction_energy(code_t J, int k, double* charges);
__device__ double term(double beta, int p, int k, code_t* branches, degree_t* degrees, double* charges);

__global__ void kernel(
    global_params_t* params,
    const size_t batch_size,
    const size_t k,
    code_t* branches_arr,
    degree_t* degrees_arr,
    float* output_matrix);