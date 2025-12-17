#pragma once

#include <vector>
#include <iostream>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <string>


typedef std::vector<uint8_t> prufer_t;
typedef std::vector<prufer_t> prufer_arr_t;

typedef uint16_t code_t;
typedef uint8_t label_size_t;
typedef uint8_t degree_t;

typedef std::unordered_map<code_t, degree_t> tree_map_t;

typedef struct global_params {
    std::vector<int> primes;
    std::vector<double> charges;
    int* gpu_primes_ptr;
    double* gpu_charges_ptr;
    float beta_step;
    float beta_c;

    std::mutex perm_mutex;
    size_t permutations;
} global_params_t;

template <typename T>
std::vector<T> extract_array(char* str) {
  std::vector<T> arr;

  // CITATION: https://stackoverflow.com/a/26228023 and `man strsep`
  char* token;
  while ((token = strsep(&str, ","))) {
    T n = std::stod(token);
    if (n != (T) 0) arr.push_back(n);
  }

  return arr;
}

template <typename T>
void print_vector(std::vector<T> v) {
  for (T entry : v) {
    std::cout << std::to_string(entry) << ",";
  }
  std::cout << std::endl;
}