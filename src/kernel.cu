#include "kernel.h"

/**
 * Transfers cpu tree data to gpu and invokes gpu kernel
 *
 * \param params  pointer to the cpu copy of the global params struct
 * \param gpu_params pointer to the gpu copy of the global params struct
 * \param batch_size number of trees being computed
 * \param k number of particles/charges
 * \param branches_arr buffer of branch data batch
 * \param degrees_arr buffer of degree data batch
 * \param gpu_output_matrix pointer to output matrix on the gpu
 */
void compute_batch(global_params_t* params, global_params_t* gpu_params, const size_t batch_size, const size_t k, code_t* branches_arr, degree_t* degrees_arr, float* gpu_output_matrix) {
  //declares gpu tree data buffers
  code_t* gpu_branches_arr;
  degree_t* gpu_degrees_arr;

  size_t branches_bytes = sizeof(code_t)*batch_size*k;
  size_t degrees_bytes = sizeof(degree_t)*batch_size*k;

  //allocates and copies tree data buffers to gpu
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

  // Invocation of gpu kernel
  dim3 blockDims(params->primes.size(), params->beta_count);
  kernel<<<blockDims, batch_size>>>(gpu_params, batch_size, k, gpu_branches_arr, gpu_degrees_arr, gpu_output_matrix);

  //Synchronize the gpu 
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
  }
}

/**
 * Computes the falling factorial (x)_n. See more https://en.wikipedia.org/wiki/Falling_and_rising_factorials
 *
 * \param x integer to take factorial of
 * \param n degree of factorial
 */
__device__ int falling_factorial(int x, int n) {
  int prod = 1;
  for (int i = x-n+1; i <= x; i++) {
    prod *= i;
  }
  return prod;
}

/**
 * Computes interaction energy of specified subset of charges array. Interaction energy is defined as $e_J = \sum_{i,j \in J and i < j} q_iq_j$.
 *
 * \param J encoded subset of charges array to consider
 * \param k number of particles/charges
 * \param charges array of charges of particles
 */
__device__ double interaction_energy(code_t J, int k, double* charges) {
  double sum = 0.0;
  for (size_t i = 0; i < k-1; i++) {
    for (size_t j = i+1; j < k; j++) {
      //if the ith and jth bit are in J, then add the product of their charges to the sum
      if (((1UL << i) & J) && ((1UL << j) & J)) 
        sum += charges[i] * charges[j];
    }
  }
  return sum;
}

/**
 * Computes summand that tree described by `branches` and `degrees` contributes to Z_N(\beta).
 *
 * \param beta inverse temperature parameter
 * \param p p-adic space to consider
 * \param k number of particles/charges
 * \param branches set of branches of tree
 * \param degrees splitting degree for each branch
 * \param charges array of charges of particles
 */
__device__ double term(double beta, int p, int k, code_t* branches, degree_t* degrees, double* charges) {
  double prod = 1.0;

  for (int i = 0; i < k; i++) {
    code_t J = branches[i];
    if (J != 0) {
      //CITATION: https://stackoverflow.com/questions/24060025/cuda-programming-bitwise-count-rowwise
      //With CUDA, to count the bits in a number, you need to use __popc().
      prod *= falling_factorial(p, degrees[i]) / 
              (pow(p, __popc(J)+interaction_energy(J, k, charges)*beta) - p);
    }
  }

  return prod;
}


/**
 * Transfers cpu tree data to gpu and invokes gpu kernel
 *
 * \param gpu_params pointer to the gpu copy of the global params struct
 * \param batch_size number of trees being computed
 * \param k number of particles/charges
 * \param branches_arr buffer of branch data batch
 * \param degrees_arr buffer of degree data batch
 * \param gpu_output_matrix pointer to output matrix on the gpu
 */
__global__ void kernel(global_params_t* gpu_params, const size_t batch_size, const size_t k, code_t* branches_arr, degree_t* degrees_arr, float* gpu_output_matrix) {
  int prime = gpu_params->gpu_primes_ptr[blockIdx.x];
  double beta = blockIdx.y * gpu_params->beta_step;

  // Gets address of current branches array in contiguous buffer;
  code_t* branches = &branches_arr[threadIdx.x * k];
  degree_t* degrees = &degrees_arr[threadIdx.x * k];
  
  //compute summand of Z_N(\beta)
  float summand = term(beta, prime, k, branches, degrees, gpu_params->gpu_charges_ptr);

  //add summand to current value of that partition function  
  int partition_idx = blockIdx.y * gridDim.x + blockIdx.x;
  atomicAdd((float*) &gpu_output_matrix[partition_idx], summand);
}