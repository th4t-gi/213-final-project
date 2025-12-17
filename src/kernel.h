#pragma once

#include "utils.h"

void prufer_seq_to_tree(int N, prufer_t seq, std::vector<code_t>& branches, std::vector<degree_t>& degrees);

void compute_batch(global_params_t* params, prufer_arr_t permutations, size_t perm_batch_size);

__global__ void kernel(
    global_params_t* params,
    const size_t batch_size,
    const size_t k,
    code_t** branches_arr,
    degree_t** degrees_arr,
    double** partition_values);