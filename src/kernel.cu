#include "kernel.h"
#include <algorithm>



void prufer_seq_to_tree(int N, prufer_t seq, code_t branches[], degree_t degrees[]) {
  int prufer_max = *std::max_element(seq.begin(), seq.end());

  // add implied edge to prufer sequence
  seq.push_back(1);
  
  //https://stackoverflow.com/questions/13110130/initialize-a-vector-to-zeros-c-c11
  // branches = std::vector<code_t>(N-1, 0);
  // degrees = std::vector<degree_t>(N-1);

  for (size_t i = 0; i < seq.size(); i++) {
    uint8_t branch_index = seq[i] - 1;
    // for each leave index, add the binary encoded value to the branch index
    if (i < N) {
      // std::cout << "i: " << +branch_index << " -- " << i << " and " << (2<<i) << std::endl;
      branches[branch_index] += (1 << i);
    } else {
      // for indecies in the prufer seq that refer to branches, find index of parent and add its sum to parent
      branches[branch_index] += branches[prufer_max + N - i - 1];
    }
  }

  // count appearances of each index for the degrees of each branch
  for (int i = 0; i < prufer_max; i++) {
    degree_t d = std::count(seq.begin(), seq.end(), i+1);
    degrees[i] = d;
  }
}


void compute_batch(global_params_t* params, prufer_arr_t permutations, const size_t batch_size) {

}

void worker(int k, prufer_t Sk, global_params_t* params, const size_t batch_size) {
  // prufer_arr_t perms_of_Sk = {};
  code_t branches_arr[batch_size][k];
  degree_t degrees_arr[batch_size][k];

  global_params_t* gpu_params;

  if (cudaMalloc(&params->gpu_charges_ptr, sizeof(int)*k) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMalloc(&params->gpu_primes_ptr, sizeof(int)*params->primes.size()) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMalloc(&gpu_params, sizeof(global_params_t)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2);
  }
  if (cudaMemcpy(params->gpu_charges_ptr, params->charges.data(), sizeof(int)*k, cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }
  if (cudaMemcpy(params->gpu_primes_ptr, params->primes.data(), sizeof(int)*params->primes.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }
  if (cudaMemcpy(gpu_params, params, sizeof(global_params_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
    exit(2);
  }

  long n = 0;
  size_t batch_count = 0;

  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  do {
    //Checks if permutation is a valid phylogenetic tree NOT CORRECT
    code_t* branches = branches_arr[batch_count];
    degree_t* degrees = degrees_arr[batch_count];
    for (int i = 0; i < k; i++) {
      branches[i] = 0;
      degrees[i] = 0;
    }
    prufer_seq_to_tree(k, Sk, branches, degrees);
    bool is_valid_tree = true;
    //https://stackoverflow.com/questions/22880431/iterate-through-unordered-map-c
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
      // perms_of_Sk.push_back(Sk);
      // print_vector_u(Sk);
      // for (auto entry : tree) {
      //   std::cout << entry.first << ":" << +entry.second << ", ";
      // }
      // std::cout << std::endl;

      // std::cout << "CORRECT: "; print_vector_u(Sk);
      // for (int i = 0; i < k; i++) {
      //   std::cout << branches[i] << ",";
      // }
      // std::cout << std::endl;
      batch_count++;
    } else {
      // std::cout << "invalid perm: "; print_vector_u(Sk);
      // for (int i = 0; i < k; i++) {
      //   std::cout << branches[i] << ",";
      // }
      // std::cout << std::endl;
    }

    
    if (batch_count == batch_size) {
      n += batch_size;

      int beta_count = ceil(params->beta_c/params->beta_step);
      int n_primes = params->primes.size();

      code_t** gpu_branches;
      degree_t** gpu_degrees;
      double** gpu_partition_values;

      //TODO: Send off to gpu
      if (cudaMalloc(&gpu_branches, sizeof(code_t)*batch_size*k) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate branches array on GPU");
        exit(2);
      }
      if (cudaMalloc(&gpu_degrees, sizeof(degree_t)*batch_size*k) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate degrees array on GPU");
        exit(2);
      }
      if (cudaMalloc(&gpu_partition_values, sizeof(double)*n_primes*beta_count) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate partition values array on GPU");
        exit(2);
      }
      if (cudaMemcpy(gpu_branches, branches_arr, sizeof(code_t)*batch_size*k, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to branches array to the GPU.");
        exit(2);
      }
      if (cudaMemcpy(gpu_degrees, degrees_arr, sizeof(degree_t)*batch_size*k, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to degrees array to the GPU.");
        exit(2);
      }

      dim3 inputDims(n_primes, beta_count);
      kernel<<<inputDims, batch_size>>>(gpu_params, batch_size, k, gpu_branches, gpu_degrees, gpu_partition_values);

      if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }

      batch_count = 0;
    }
  } while (std::next_permutation(Sk.begin(), Sk.end()));

  // compute_batch();

  n += batch_count;
  std::cout << "worker checked " << n << " permutations." << std::endl;

  params->perm_mutex.lock();
  params->permutations += n;
  params->perm_mutex.unlock();
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


__global__ void kernel(global_params_t* params, const size_t batch_size, const size_t k, code_t** branches_arr, degree_t** degrees_arr, double** partition_values) {

  int prime = params->gpu_primes_ptr[blockIdx.x];

  double beta = blockIdx.y * params->beta_step;
  // printf("prime: %d, beta: %lf\n", prime, beta);

  code_t* branches = branches_arr[threadIdx.x];
  degree_t* degrees = degrees_arr[threadIdx.x];


  printf("threadIdx: %d\n", threadIdx.x);
  printf("branches: %p\n", branches_arr[0]);//, branches[0]);
  
  // printf("prime: %d, beta: %f, batch_size: %lu, k: %lu\n", prime, beta, batch_size, k);

  // for (int i = 0; i < k; i++) {
  //   printf("%d ", branches[i]);
  // }

  partition_values[blockIdx.y][blockIdx.x] += 1.0;
  // partition_values[blockIdx.x][blockIdx.y] = term(beta, prime, k, branches, degrees, params->gpu_charges_ptr);
}