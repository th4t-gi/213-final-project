#include "kernel.h"
#include <algorithm>

void compute_batch(global_params_t* params, global_params_t* gpu_params, const size_t batch_size, const size_t k, code_t* branches_arr, degree_t* degrees_arr, float* gpu_partition_values) {
  code_t* gpu_branches_arr;
  degree_t* gpu_degrees_arr;

  size_t branches_bytes = sizeof(code_t)*batch_size*k;
  size_t degrees_bytes = sizeof(degree_t)*batch_size*k;

  if (cudaMalloc(&gpu_branches_arr, branches_bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate branches array on GPU");
    exit(2);
  }
  if (cudaMalloc(&gpu_degrees_arr, degrees_bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate degrees array on GPU");
    exit(2);
  }
  if (cudaMemcpy(gpu_branches_arr, branches_arr, branches_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to branches array to the GPU.");
    exit(2);
  }
  if (cudaMemcpy(gpu_degrees_arr, degrees_arr, degrees_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to degrees array to the GPU.");
    exit(2);
  }

  dim3 inputDims(params->primes.size(), params->beta_count);
  kernel<<<inputDims, batch_size>>>(gpu_params, batch_size, k, gpu_branches_arr, gpu_degrees_arr, gpu_partition_values);

  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
}


__device__ int falling_factorial(int x, int k) {
  int prod = 1;
  for (int i = x-k+1; i <= x; i++) {
    prod *= i;
  }
  return prod;
}

__device__ double interaction_energy(int k, code_t J, double* charges) {
  double sum = 0.0;
  for (size_t i = 0; i < k-1; i++) {
    for (size_t j = i+1; j < k; j++) {
      if (i & J && j & J) sum += charges[i] * charges[j];
    }
  }
  return sum;
}

__device__ double term(double beta, int p, int k, code_t* branches, degree_t* degrees, double* charges) {
  double prod = 1.0;

  for (int i = 0; i < k; i++) {
    code_t J = branches[i];
    if (J != 0) {
      //https://stackoverflow.com/questions/24060025/cuda-programming-bitwise-count-rowwise
      prod *= falling_factorial(p, degrees[i]) / 
              (pow(p, __popc(J)+interaction_energy(k, J, charges)*beta) - p);
    }
  }

  return prod;
}


__global__ void kernel(global_params_t* params, const size_t batch_size, const size_t k, code_t* branches_arr, degree_t* degrees_arr, float* partition_values) {

  int prime = params->gpu_primes_ptr[blockIdx.x];

  double beta = blockIdx.y * params->beta_step;
  // printf("prime: %d, beta: %lf\n", prime, beta);

  // Gets address of current branches array in contiguous buffer;
  code_t* branches = &branches_arr[threadIdx.x * k];
  degree_t* degrees = &degrees_arr[threadIdx.x * k];
  
  //compute summand of Z_N(\beta)
  float summand = term(beta, prime, k, branches, degrees, params->gpu_charges_ptr);

  // if (prime == 2 && beta < 0.15) {
  // }

  //add summand to current value of that partition function  
  int partition_idx = blockIdx.x * blockDim.x + blockIdx.y;
  // printf("p: %d, b: %lf, idx: %d, x: %d, y: %d, blockdim: %d, summand: %f\n", prime, beta, partition_idx, blockIdx.x, blockIdx.y, blockDim.x, summand);
  atomicAdd((float*) &partition_values[partition_idx], summand);
  // partition_values[partition_idx] += summand;

  // if (prime == 2 && beta < 0.15) {
  //   printf("@p: %d, b: %lf, Z_I(b): %lf\n", prime, beta, partition_values[partition_idx]);
  // }
}