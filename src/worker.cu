#include "worker.h"

/**
 * worker function for each Thread
 *
 * \param k number of charges/leaves in each tree
 * \param Sk the prufer "set" to permute
 * \param params global parameters struct
 * \param batch_size maximum number of trees to send over to gpu at one time
 */
void worker(int k, prufer_t Sk, global_params_t* params, const size_t batch_size) {
  // declares cpu tree data buffers
  code_t branches_arr[batch_size*k];
  degree_t degrees_arr[batch_size*k];

  size_t charges_bytes = sizeof(double)*k;
  size_t primes_bytes = sizeof(int)*params->primes.size();
  size_t partition_values_bytes = sizeof(float)*params->primes.size()* params->beta_count;

  //declares gpu buffers for params and output matrix
  global_params_t* gpu_params;
  float* gpu_partition_values;
  //allocates for thread_specific output matrix
  float* cpu_partition_values = (float*) malloc(partition_values_bytes);

  //Transfers all global parameter data over to gpu
  if (cudaMalloc(&params->gpu_charges_ptr, charges_bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMemcpy(params->gpu_charges_ptr, params->charges.data(), charges_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }
  if (cudaMalloc(&params->gpu_primes_ptr, primes_bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMemcpy(params->gpu_primes_ptr, params->primes.data(), primes_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }
  if (cudaMalloc(&gpu_params, sizeof(global_params_t)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMemcpy(gpu_params, params, sizeof(global_params_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }
  if (cudaMalloc(&gpu_partition_values, partition_values_bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate partition values array on GPU");
    exit(2);
  }


  long n = 0;
  size_t batch_count = 0;

  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  // Loops over all perumtations of Sk and validates trees
  do {
    //extracts pointers to current position in branches/degrees buffers and zeros out
    code_t* branches = &branches_arr[batch_count*k];
    degree_t* degrees = &degrees_arr[batch_count*k];
    for (int i = 0; i < k; i++) {
      branches[i] = 0;
      degrees[i] = 0;
    }
    prufer_seq_to_tree(k, Sk, branches, degrees);
    bool is_valid_tree = true;

    //Checks if permutation is a valid phylogenetic tree by checking that the first element is 2^k - 1 and that all branches are in descending order
    //CITATION: Ian Clawson helped me figure out these criteria to correctly validate trees.
    if (__builtin_popcount(branches[0]) != k) {
      is_valid_tree = false;
      continue;
    }
    for (int i = 0; i < k-1; i++) {
      if (branches[i] < branches[i+1]) {
        is_valid_tree = false;
        break;
      }
    }

    // incrementing batch_count is equivalent to "saving" tree to database because it doesn't get overwritten on next iteration
    if (is_valid_tree) {
      batch_count++;
    }
    
    // If batch is full, send it off and reset buffers
    if (batch_count == batch_size) {
      n += batch_size;

      //Send batch off
      compute_batch(params, gpu_params, batch_count, k, branches_arr, degrees_arr, gpu_partition_values);

      //zero out whole buffers
      for (int i = 0; i < batch_size*k; i++) {
        branches_arr[i] = 0;
        degrees_arr[i] = 0;
      }
      batch_count = 0;
    }
  } while (std::next_permutation(Sk.begin(), Sk.end()));

  //send off final batch (batch_count < batch_size)
  n += batch_count;
  compute_batch(params, gpu_params, batch_count, k, branches_arr, degrees_arr, gpu_partition_values);


  // Copy partition values data back from GPU
  if (cudaMemcpy(cpu_partition_values, gpu_partition_values, partition_values_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to copy partition values back from the GPU\n");
  }

  //CITATION: https://www.geeksforgeeks.org/cpp/std-mutex-in-cpp/
  // Locking and adding all values from cpu output matrix buffer to global output matrix
  params->output_mutex.lock();
  params->permutations += n;
  for (int i = 0; i < params->primes.size()* params->beta_count; i++) {
    params->partition_values[i] += cpu_partition_values[i];
  }
  params->output_mutex.unlock();

  std::cout << "worker computed " << n << " permutations." << std::endl;

  free(cpu_partition_values);
  cudaFree(gpu_partition_values);
}
