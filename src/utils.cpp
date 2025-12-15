#include "utils.h"
#include <algorithm>

std::vector<int> extract_array(char* str) {
  std::vector<int> arr;

  // CITATION: https://stackoverflow.com/a/26228023 and `man strsep`
  char* token;
  while ((token = strsep(&str, ","))) {
    int n = atoi(token);
    if (n != 0) arr.push_back(n);
  }

  return arr;
}

void print_vector_u(std::vector<uint8_t> v) {
//   std::cout << "hello?" << v.size() << std::endl;
  for (uint8_t entry : v) {
    printf("%d,", entry);
  }
  std::cout << std::endl;
}

void print_vector(std::vector<int> v) {
  for (int entry : v) {
    std::cout << entry << ",";
  }
  std::cout << std::endl;
}
void print_vector(std::vector<uint16_t> v) {
  for (uint16_t entry : v) {
    std::cout << entry << ",";
  }
  std::cout << std::endl;
}

tree_map_t prufer_seq_to_branches(int N, prufer_t seq) {
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
  // print_vector(branches);

  // for (int i = 0; i < seq.size(); i++) {
  //   uint8_t branch_index = seq[i] - 1;
  //   // for each leave index, add the binary encoded value to the branch index
  //   if (i >= N) {
  //   }
  // }


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
