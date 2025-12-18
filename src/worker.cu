#include "worker.h"

void worker(int k, prufer_t Sk, global_params_t* params, const size_t batch_size) {
  // prufer_arr_t perms_of_Sk = {};
  code_t branches_arr[batch_size*k];
  degree_t degrees_arr[batch_size*k];

  size_t charges_bytes = sizeof(double)*k;
  size_t primes_bytes = sizeof(int)*params->primes.size();
  size_t partition_values_bytes = sizeof(float)*params->primes.size()* params->beta_count;

  global_params_t* gpu_params;
  float* gpu_partition_values;
  float* cpu_partition_values = (float*) malloc(partition_values_bytes);

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
  // if (cudaMemset(&gpu_partition_values, 0, partition_values_bytes) != cudaSuccess) {
  //   fprintf(stderr, "Failed to set partition values on GPU");
  //   exit(2);
  // };


  long n = 0;
  size_t batch_count = 0;

  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  do {
    //Checks if permutation is a valid phylogenetic tree
    code_t* branches = &branches_arr[batch_count*k];
    degree_t* degrees = &degrees_arr[batch_count*k];
    for (int i = 0; i < k; i++) {
      branches[i] = 0;
      degrees[i] = 0;
    }
    prufer_seq_to_tree(k, Sk, branches, degrees);
    bool is_valid_tree = true;

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

    if (is_valid_tree) {
      batch_count++;
    }
    
    if (batch_count == batch_size) {
      n += batch_size;

      //TODO: Send off to gpu
      compute_batch(params, gpu_params, batch_size, k, branches_arr, degrees_arr, gpu_partition_values);

      // printf("sycronized!\n");

      for (int i = 0; i < batch_size*k; i++) {
        branches_arr[i] = 0;
        degrees_arr[i] = 0;
      }
      batch_count = 0;
    }
  } while (std::next_permutation(Sk.begin(), Sk.end()));

  n += batch_count;
  compute_batch(params, gpu_params, batch_size, k, branches_arr, degrees_arr, gpu_partition_values);


  // Copy partition values data back from GPU
  if (cudaMemcpy(cpu_partition_values, gpu_partition_values, partition_values_bytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    fprintf(stderr, "Failed to copy partition values back from the GPU\n");
  }

  // for (int i = 0; i < params->primes.size(); i++) {
  //   for (int j = 0; j < params->beta_count; j++) {
  //     int idx = i*params->primes.size() + j;
  //     printf("~p: %d, b: %lf, Z_I(b): %lf\n", params->primes[i], params->beta_step*j, cpu_partition_values[idx]);
  //   }
  // }

  params->output_mutex.lock();
  params->permutations += n;
  for (int i = 0; i < params->primes.size()* params->beta_count; i++) {
    params->partition_values[i] += cpu_partition_values[i];
  }
  params->output_mutex.unlock();

  std::cout << "worker checked " << n << " permutations." << std::endl;

  //TODO: free cpu_partition_values
  free(cpu_partition_values);
  cudaFree(gpu_partition_values);
}
