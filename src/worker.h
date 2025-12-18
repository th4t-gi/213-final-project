#pragma once
#include "utils.h"
#include "kernel.h"

void worker(int k, prufer_t Sk, global_params_t* params, size_t batch_size);