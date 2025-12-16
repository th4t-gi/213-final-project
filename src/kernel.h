#pragma once
#include "utils.h"

tree_map_t prufer_seq_to_tree(int N, prufer_t seq);

void compute_batch(global_params_t* params, prufer_arr_t permutations, size_t perm_batch_size);

void worker(int k, prufer_t Sk);

void kernel(global_params_t config, prufer_arr_t permutations, size_t perm_batch_size);