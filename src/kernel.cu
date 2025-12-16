#include "kernel.h"
#include <algorithm>


__host__ __device__ tree_map_t prufer_seq_to_tree(int N, prufer_t seq) {
  int prufer_max = *std::max_element(seq.begin(), seq.end());

  // add implied edge to prufer sequence
  seq.push_back(1);
  
  //https://stackoverflow.com/questions/13110130/initialize-a-vector-to-zeros-c-c11
  std::vector<code_t> branches(prufer_max, 0);

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

  tree_map_t tree;
  tree.reserve(prufer_max);
  // count appearances of each index for the degrees of each branch
  for (int i = 0; i < branches.size(); i++) {
    int J = branches[i];
    int degrees = std::count(seq.begin(), seq.end(), i+1);
    tree[J] = degrees;
  }

  return tree;
}


__host__ void compute_batch(global_params_t* params, prufer_arr_t permutations, size_t perm_batch_size) {

}

__host__ void worker(int k, prufer_t Sk, global_params_t* params, size_t perm_batch_size) {
  prufer_arr_t perms_of_Sk = {};
  perms_of_Sk.reserve(perm_batch_size);

  global_params_t* gpu_params;

  if (cudaMalloc(&gpu_params, sizeof(global_params_t)) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate params struct on GPU");
    exit(2)
  }
  if (cudaMemcpy(gpu_params, params, sizeof(global_params_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    fprintf(stderr, "Failed to coppy boards to the GPU.");
  }

  long n = 0;

  // CITATION: https://www.geeksforgeeks.org/cpp/stdnext_permutation-prev_permutation-c/
  do {
    //Checks if permutation is a valid phylogenetic tree NOT CORRECT
    // tree_map_t tree = prufer_seq_to_branches(k, Sk);
    bool is_valid_tree = true;
    //https://stackoverflow.com/questions/22880431/iterate-through-unordered-map-c
    // for (auto entry : tree) {
    //   if (__builtin_popcount(entry.first) == 1) {
    //     is_valid_tree = false;
    //     break;
    //   }
    // }

    if (is_valid_tree) {
      perms_of_Sk.push_back(Sk);
      // print_vector_u(Sk);
      // for (auto entry : tree) {
      //   std::cout << entry.first << ":" << +entry.second << ", ";
      // }
      // std::cout << std::endl;
    }

    
    if (perms_of_Sk.size() == perm_batch_size) {
      n += perms_of_Sk.size();
      //TODO: Send off to gpu

      double** partition_values;
      
      int beta_count = ceil(params->beta_c/params->beta_step);
      dim3 inputDims(params->primes.size(), beta_count);
      kernel<<<inputDims, perm_batch_size>>>(gpu_params, perms_of_Sk, GPU_BATCH_SIZE);

      if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
      }




      perms_of_Sk.clear();
    }
  } while (std::next_permutation(Sk.begin(), Sk.end()));

  n += perms_of_Sk.size();
  std::cout << "worker completed " << n << std::endl;

  // perm_mtx.lock();
  // permutations+= n;
  // perm_mtx.unlock();
}


__global__ void kernel(global_params_t* params, prufer_arr_t permutations, size_t perm_batch_size) {

  int prime = params->primes[blockIdx.x];
  double beta = blockIdx.y * params->beta_step;

  printf("prime: %d, beta: %f\n", prime, beta)

}